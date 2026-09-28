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

    def dot_row(self, row: Int, x: FPtr) -> Float32:
        """Returns the dot product of row `row` with x[cols]."""
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

    def dot_row(self, row: Int, x: FPtr) -> Float32:
        var d = F32V(0)
        for i in range(0, self.cols, NW):
            d = self.load[NW](row, i).fma(x.unsafe_load[width=NW](i), d)
        return d.reduce_add()

    def nbytes(self) -> Int:
        return self.rows * self.cols * size_of[Scalar[Self.dtype]]()

    def free(self):
        self.data.unsafe_free()


struct QuantMatrix[BITS: Int, GROUP: Int, SYMMETRIC: Bool](WeightMatrix):
    """Affine-quantized weights: BITS-bit codes with a scale and an integer
    zero point per group of GROUP weights along the reduction axis.

    Each weight is stored as an unsigned code u in [0, 2^BITS - 1] and read
    back as

        w = scale * (u - zero_point)

    the standard affine form (as in PyTorch, ONNX and TFLite). Per group,
    `scale` is a float16 and `zero_point` an unsigned integer code in the same
    range as u, so w = 0 is represented exactly, by u = zero_point. GROUP = 0
    means one group spans the whole reduction axis ("per-channel": one pair
    per output column of a layer matrix, per vocabulary row of wte).

    - Asymmetric: the group's range, widened to include 0, is mapped onto
      [0, 2^BITS - 1]: scale = (max - min) / (2^BITS - 1) and
      zero_point = round(-min / scale).
    - Symmetric: zero_point = 2^(BITS-1) (8 for int4, 128 for int8) and
      scale = max|w| / (2^(BITS-1) - 1), so u - zero_point is a signed code
      centered on zero.

    In a dot product the zero point factors out of each group:
    sum(x * w) = scale * (sum(x * u) - zero_point * sum(x)), which keeps the
    inner sum in integers for the integer kernel (M4).

    Rounding is plain round-to-nearest. int4 packs two codes per byte along
    each row: byte k holds column 2k in its low 4 bits, 2k+1 in its high 4.
    """

    comptime OUT_MAJOR = False
    comptime LEVELS = (1 << Self.BITS) - 1  # largest code
    comptime HALF = 1 << (Self.BITS - 1)  # the zero point when symmetric

    var data: Pointer[UInt8, MutUntrackedOrigin]  # the codes u
    var scale: Pointer[Float16, MutUntrackedOrigin]  # one per group
    var zero_point: Pointer[UInt8, MutUntrackedOrigin]  # one per group
    var rows: Int
    var cols: Int
    var group: Int  # GROUP, or the reduction axis length when GROUP == 0
    var reduce_rows: Bool

    def __init__(
        out self,
        data: Pointer[UInt8, MutUntrackedOrigin],
        scale: Pointer[Float16, MutUntrackedOrigin],
        zero_point: Pointer[UInt8, MutUntrackedOrigin],
        rows: Int,
        cols: Int,
        group: Int,
        reduce_rows: Bool,
    ):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
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
        var zero_point = unsafe_alloc[UInt8](ngroups)
        var m = Self(data, scale, zero_point, rows, cols, group, reduce_rows)

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

            var lo = Float32(0)  # the range always includes 0
            var hi = Float32(0)
            for i in idx:
                lo = min(lo, src[unsafe_offset=i])
                hi = max(hi, src[unsafe_offset=i])
            var s: Float32
            comptime if Self.SYMMETRIC:
                s = max(-lo, hi) / Float32(Self.HALF - 1)
            else:
                s = (hi - lo) / Float32(Self.LEVELS)
            if s == 0:  # an all-zero group: any scale reproduces it exactly
                s = 1
            # Quantize against the float16-rounded scale actually stored.
            var s16 = s.cast[DType.float16]()
            var sf = s16.cast[DType.float32]()
            var zp: Int
            comptime if Self.SYMMETRIC:
                zp = Self.HALF
            else:
                zp = clamp_code(Int(round(-lo / sf)), Self.LEVELS)
            scale[unsafe_offset=gi] = s16
            zero_point[unsafe_offset=gi] = UInt8(zp)
            for i in idx:
                var u = Int(round(src[unsafe_offset=i] / sf)) + zp
                m.set_code(i, clamp_code(u, Self.LEVELS))
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
            # Groups run down the rows: each column has its own scale and
            # zero point, so load `width` of each.
            var g = (row // self.group) * self.cols + col
            var s = self.scale.unsafe_load[width=width](g).cast[DType.float32]()
            var zp = self.zero_point.unsafe_load[width=width](g).cast[
                DType.float32
            ]()
            return (u - zp) * s
        else:
            # Groups run along the row: `width` columns share one scale and
            # zero point (width divides the group size).
            var g = (row * self.cols + col) // self.group
            var s = self.scale[unsafe_offset=g].cast[DType.float32]()
            var zp = self.zero_point[unsafe_offset=g].cast[DType.float32]()
            return (u - SIMD[DType.float32, width](zp)) * SIMD[
                DType.float32, width
            ](s)

    def dequant_row(self, row: Int, dst: FPtr):
        for i in range(0, self.cols, NW):
            dst.unsafe_store(i, self.load[NW](row, i))

    def dot_row(self, row: Int, x: FPtr) -> Float32:
        var d = F32V(0)
        for i in range(0, self.cols, NW):
            d = self.load[NW](row, i).fma(x.unsafe_load[width=NW](i), d)
        return d.reduce_add()

    def nbytes(self) -> Int:
        """Codes, plus a float16 scale and a one-byte zero point per group."""
        var ngroups = self.rows * self.cols // self.group
        return self.rows * self.cols * Self.BITS // 8 + ngroups * 3

    def free(self):
        self.data.unsafe_free()
        self.scale.unsafe_free()
        self.zero_point.unsafe_free()


@always_inline
def clamp_code(u: Int, levels: Int) -> Int:
    """Clamps a code to [0, levels]."""
    return max(0, min(levels, u))
