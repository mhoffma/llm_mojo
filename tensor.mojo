"""Weight storage formats.

A weight matrix is always *used* as float32, but it can be *stored* however a
format likes: float32, float16, quantized integers, ... Each format is a
struct implementing the `WeightMatrix` trait, and the kernels in kernels.mojo
are generic over that trait.

Mojo has no class inheritance. A trait is an interface, and a function or
struct generic over it (`def f[W: WeightMatrix](w: W)`) is compiled separately
for each concrete format, with the format's methods inlined. So there is no
virtual-call cost in the inner loops: `load` on a float32 matrix compiles to
exactly the plain SIMD load the untyped gpt2.mojo does.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of, size_of
from std.os import abort
from std.math import round

comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime NW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, NW]


trait WeightMatrix(Deinitable, ImplicitlyCopyable):
    """A [rows, cols] matrix of weights, read back as float32.

    Values are handles: copying one copies the pointer, not the data, and the
    owner releases the storage by calling `free` once. That keeps them cheap to
    pass to kernels and to store in a List.
    """

    comptime OUT_MAJOR: Bool
    """Storage orientation of the layer matrices. False: [IN, OUT], as in
    Hugging Face's GPT-2 (kernels.matmul). True: [OUT, IN], one output per row,
    as in GGUF files (kernels.matmul_rows)."""

    @staticmethod
    def name() -> String:
        """Short name for the format, as used by --dtype."""
        ...

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self:
        """Converts a row-major float32 matrix into this format (a copy).

        `reduce_rows` says which axis the kernels sum over: True for the layer
        matrices ([IN, OUT], x @ W sums down the rows), False for wte ([V, C],
        the output head sums along each row). Quantized formats group their
        scales along that axis; others ignore it.
        """
        ...

    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """Returns `width` consecutive values of `row`, starting at `col`.

        `col` must be a multiple of `width`.
        """
        ...

    def dequant_row(self, row: Int, dst: FPtr):
        """Writes row `row` (all `cols` values) to dst as float32."""
        ...

    def nbytes(self) -> Int:
        """Bytes of storage, to report the model's size."""
        ...

    def free(self):
        """Releases the storage."""
        ...


struct DenseMatrix[dtype: DType](WeightMatrix):
    """Every element stored as `dtype` (float32, float16, bfloat16, ...).

    `load` widens to float32 with `.cast`, which is free for float32, one
    VCVTPH2PS for float16, and a shift for bfloat16 (see
    research/test_half.mojo).
    """

    @staticmethod
    def name() -> String:
        if Self.dtype == DType.float32:
            return "f32"
        if Self.dtype == DType.float16:
            return "f16"
        return "bf16"

    comptime OUT_MAJOR = False

    var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        rows: Int,
        cols: Int,
    ):
        self.data = data
        self.rows = rows
        self.cols = cols

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self:
        var n = rows * cols
        var data = unsafe_alloc[Scalar[Self.dtype]](n)
        for i in range(0, n - n % NW, NW):
            data.unsafe_store(
                i, src.unsafe_load[width=NW](i).cast[Self.dtype]()
            )
        for i in range(n - n % NW, n):
            data[unsafe_offset=i] = src[unsafe_offset=i].cast[Self.dtype]()
        return Self(data, rows, cols)

    @always_inline
    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        return self.data.unsafe_load[width=width](row * self.cols + col).cast[
            DType.float32
        ]()

    def dequant_row(self, row: Int, dst: FPtr):
        for i in range(0, self.cols, NW):
            dst.unsafe_store(i, self.load[NW](row, i))

    def nbytes(self) -> Int:
        return self.rows * self.cols * size_of[Scalar[Self.dtype]]()

    def free(self):
        self.data.unsafe_free()


struct QuantMatrix[BITS: Int, GROUP: Int, SYMMETRIC: Bool](WeightMatrix):
    """Affine-quantized weights: BITS-bit codes with a scale and offset per
    group of GROUP weights along the reduction axis.

    Each weight is stored as an unsigned code u in [0, 2^BITS - 1] and read
    back as

        w = scale * u + min

    with one (scale, min) pair per group, stored as float16. GROUP = 0 means
    one group spans the whole reduction axis ("per-channel": one pair per
    output column of a layer matrix, per vocabulary row of wte).

    - Asymmetric: min and scale fit the group's [min, max] exactly.
    - Symmetric: centered on zero, scale = max|w| / (2^(BITS-1) - 1) and
      min = -2^(BITS-1) * scale, so u - 2^(BITS-1) is a signed code. This is
      the form the integer kernel (M4) wants.

    Rounding is plain round-to-nearest. int4 packs two codes per byte along
    each row: byte k holds column 2k in its low 4 bits, 2k+1 in its high 4.
    """

    comptime OUT_MAJOR = False
    comptime LEVELS = (1 << Self.BITS) - 1  # largest code
    comptime HALF = 1 << (Self.BITS - 1)  # the code for zero when symmetric

    var data: Pointer[UInt8, MutUntrackedOrigin]
    var scale: Pointer[Float16, MutUntrackedOrigin]
    var minv: Pointer[Float16, MutUntrackedOrigin]
    var rows: Int
    var cols: Int
    var group: Int  # GROUP, or the reduction axis length when GROUP == 0
    var reduce_rows: Bool

    def __init__(
        out self,
        data: Pointer[UInt8, MutUntrackedOrigin],
        scale: Pointer[Float16, MutUntrackedOrigin],
        minv: Pointer[Float16, MutUntrackedOrigin],
        rows: Int,
        cols: Int,
        group: Int,
        reduce_rows: Bool,
    ):
        self.data = data
        self.scale = scale
        self.minv = minv
        self.rows = rows
        self.cols = cols
        self.group = group
        self.reduce_rows = reduce_rows

    @staticmethod
    def name() -> String:
        var g = String("ch") if Self.GROUP == 0 else "g" + String(Self.GROUP)
        var s = "int" + String(Self.BITS) + "-" + g
        return s + "-sym" if Self.SYMMETRIC else s

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self:
        comptime assert Self.BITS == 4 or Self.BITS == 8, "BITS must be 4 or 8"
        var axis = rows if reduce_rows else cols
        var group = axis if Self.GROUP == 0 else Self.GROUP
        if axis % group != 0:
            abort("group size must divide the reduction axis")
        var ngroups = rows * cols // group
        var data = unsafe_alloc[UInt8](rows * cols * Self.BITS // 8)
        var scale = unsafe_alloc[Float16](ngroups)
        var minv = unsafe_alloc[Float16](ngroups)
        var m = Self(data, scale, minv, rows, cols, group, reduce_rows)

        # Visit each group as a list of flat indices: down a column when
        # reducing rows, along a row otherwise.
        var idx = List[Int](capacity=group)
        for gi in range(ngroups):
            idx.clear()
            if reduce_rows:
                var r0 = (gi // cols) * group
                var c = gi % cols
                for r in range(r0, r0 + group):
                    idx.append(r * cols + c)
            else:
                var r = gi // (cols // group)
                var c0 = (gi % (cols // group)) * group
                for c in range(c0, c0 + group):
                    idx.append(r * cols + c)

            var lo = src[unsafe_offset=idx[0]]
            var hi = lo
            for i in idx:
                lo = min(lo, src[unsafe_offset=i])
                hi = max(hi, src[unsafe_offset=i])
            var s: Float32
            var mn: Float32
            comptime if Self.SYMMETRIC:
                s = max(abs(lo), abs(hi)) / Float32(Self.HALF - 1)
                mn = -Float32(Self.HALF) * s
            else:
                s = (hi - lo) / Float32(Self.LEVELS)
                mn = lo
            if s == 0:  # constant group: any scale reproduces it exactly
                s = 1
            # Quantize against the float16-rounded values actually stored.
            var s16 = s.cast[DType.float16]()
            var m16 = mn.cast[DType.float16]()
            scale[unsafe_offset=gi] = s16
            minv[unsafe_offset=gi] = m16
            var sf = s16.cast[DType.float32]()
            var mf = m16.cast[DType.float32]()
            for i in idx:
                var u = round((src[unsafe_offset=i] - mf) / sf)
                m.set_code(i, Int(max(Float32(0), min(Float32(Self.LEVELS), u))))
        return m

    def set_code(self, i: Int, u: Int):
        comptime if Self.BITS == 8:
            self.data[unsafe_offset=i] = UInt8(u)
        else:
            var b = self.data[unsafe_offset = i // 2]
            if i % 2 == 0:
                b = (b & 0xF0) | UInt8(u)
            else:
                b = (b & 0x0F) | UInt8(u << 4)
            self.data[unsafe_offset = i // 2] = b

    @always_inline
    def codes[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """The raw codes u for `width` columns, as float32."""
        var i = row * self.cols + col
        comptime if Self.BITS == 8:
            return self.data.unsafe_load[width=width](i).cast[DType.float32]()
        else:
            var b = self.data.unsafe_load[width = width // 2](i // 2)
            var u = (b & 0x0F).interleave(b >> 4)
            return rebind[SIMD[DType.uint8, width]](u).cast[DType.float32]()

    @always_inline
    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        var u = self.codes[width](row, col)
        if self.reduce_rows:
            # Groups run down the rows: each column has its own scale, so
            # load `width` of them.
            var g = (row // self.group) * self.cols + col
            var s = self.scale.unsafe_load[width=width](g).cast[DType.float32]()
            var mn = self.minv.unsafe_load[width=width](g).cast[DType.float32]()
            return u.fma(s, mn)
        else:
            # Groups run along the row: `width` columns share one scale
            # (width divides the group size).
            var g = (row * self.cols + col) // self.group
            var s = self.scale[unsafe_offset=g].cast[DType.float32]()
            var mn = self.minv[unsafe_offset=g].cast[DType.float32]()
            return u.fma(SIMD[DType.float32, width](s), SIMD[DType.float32, width](mn))

    def dequant_row(self, row: Int, dst: FPtr):
        for i in range(0, self.cols, NW):
            dst.unsafe_store(i, self.load[NW](row, i))

    def nbytes(self) -> Int:
        var ngroups = self.rows * self.cols // self.group
        return self.rows * self.cols * Self.BITS // 8 + ngroups * 4

    def free(self):
        self.data.unsafe_free()
        self.scale.unsafe_free()
        self.minv.unsafe_free()
