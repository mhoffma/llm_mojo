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
comptime F32x16 = SIMD[DType.float32, 16]


trait WeightMatrix(Deinitable, ImplicitlyCopyable):
    """A [rows, cols] matrix of weights, read back as float32.

    Values are handles: copying one copies the pointer, not the data, and the
    owner releases the storage by calling `free` once. That keeps them cheap to
    pass to kernels and to store in a List.
    """

    comptime OUT_MAJOR: Bool
    """Storage orientation of the layer matrices. False: [IN, OUT], as in
    Hugging Face's GPT-2 (kernels.matmul). True: [OUT, IN], one output per row,
    as in GGUF files and QuantMatrix (kernels.matmul_rows)."""

    comptime FROM_GGUF: Bool
    """True if matrices of this format are read from a GGUF file as stored
    (GGUFMatrix), False if they are converted from float32 with from_f32."""

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
    comptime FROM_GGUF = False

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
    per output of a layer matrix, per vocabulary row of wte).

    - Asymmetric: the group's range, widened to include 0, is mapped onto
      [0, 2^BITS - 1]: scale = (max - min) / (2^BITS - 1) and
      zero_point = round(-min / scale).
    - Symmetric: zero_point = 2^(BITS-1) (8 for int4, 128 for int8) and
      scale = max|w| / (2^(BITS-1) - 1), so u - zero_point is a signed code
      centered on zero.

    Layout: rows are outputs and columns the reduction axis ([OUT, IN] for
    layer matrices, which `from_f32` transposes from Hugging Face's
    [IN, OUT]; [V, C] for wte, as is). So a row's groups are contiguous, and
    the kernels use the fast OUT_MAJOR path (kernels.matmul_rows).
    int4 codes are packed per 32 weights as in llama.cpp's Q4_0: byte j holds
    weight j in its low 4 bits and weight j+16 in its high 4, so one mask and
    one shift unpack 16 weights each.

    In a dot product, the scale and zero point factor out of each group:
    sum(x * w) = scale * sum(x * (u - zero_point)). `dot_row` applies the
    scale once per group, and the integer kernel (M4) will keep the inner sum
    in integers.

    Rounding is plain round-to-nearest.
    """

    comptime OUT_MAJOR = True
    comptime FROM_GGUF = False
    comptime LEVELS = (1 << Self.BITS) - 1  # largest code
    comptime HALF = 1 << (Self.BITS - 1)  # the zero point when symmetric

    var data: Pointer[UInt8, MutUntrackedOrigin]  # the codes u
    var scale: Pointer[Float16, MutUntrackedOrigin]  # one per group
    var zero_point: Pointer[UInt8, MutUntrackedOrigin]  # one per group
    var rows: Int  # outputs
    var cols: Int  # the reduction axis
    var group: Int  # GROUP, or cols when GROUP == 0

    def __init__(
        out self,
        data: Pointer[UInt8, MutUntrackedOrigin],
        scale: Pointer[Float16, MutUntrackedOrigin],
        zero_point: Pointer[UInt8, MutUntrackedOrigin],
        rows: Int,
        cols: Int,
        group: Int,
    ):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.rows = rows
        self.cols = cols
        self.group = group

    @staticmethod
    def name() -> String:
        var g = String("ch") if Self.GROUP == 0 else "g" + String(Self.GROUP)
        var s = "int" + String(Self.BITS) + "-" + g
        return s + "-sym" if Self.SYMMETRIC else s

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self:
        comptime assert Self.BITS == 4 or Self.BITS == 8, "BITS must be 4 or 8"
        # Stored shape: n rows of k values, k along the reduction axis.
        var n = cols if reduce_rows else rows
        var k = rows if reduce_rows else cols
        var group = k if Self.GROUP == 0 else Self.GROUP
        if k % group != 0 or group % 32 != 0:
            abort("group size must divide the reduction axis and be a multiple of 32")
        var gpr = k // group  # groups per row
        var data = unsafe_alloc[UInt8](n * k * Self.BITS // 8)
        var scale = unsafe_alloc[Float16](n * gpr)
        var zero_point = unsafe_alloc[UInt8](n * gpr)
        var m = Self(data, scale, zero_point, n, k, group)

        var vals = List[Float32](length=group, fill=0)
        for r in range(n):
            for g in range(gpr):
                # Gather the group: element (r, c) of the stored matrix is
                # src[c][r] when transposing, src[r][c] otherwise.
                var lo = Float32(0)  # the range always includes 0
                var hi = Float32(0)
                for j in range(group):
                    var c = g * group + j
                    var v = src[unsafe_offset = c * cols + r] if reduce_rows else src[
                        unsafe_offset = r * cols + c
                    ]
                    vals[j] = v
                    lo = min(lo, v)
                    hi = max(hi, v)
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
                scale[unsafe_offset = r * gpr + g] = s16
                zero_point[unsafe_offset = r * gpr + g] = UInt8(zp)
                for j in range(group):
                    var u = Int(round(vals[j] / sf)) + zp
                    m.set_code(r, g * group + j, clamp_code(u, Self.LEVELS))
        return m

    def set_code(self, row: Int, col: Int, u: Int):
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        comptime if Self.BITS == 8:
            p[unsafe_offset=col] = UInt8(u)
        else:
            # Chunk col // 32; within it, weight j is in byte j % 16, in the
            # low nibble for j < 16 and the high nibble otherwise.
            var j = col % 32
            var i = (col // 32) * 16 + j % 16
            var b = p[unsafe_offset=i]
            if j < 16:
                b = (b & 0xF0) | UInt8(u)
            else:
                b = (b & 0x0F) | UInt8(u << 4)
            p[unsafe_offset=i] = b

    @always_inline
    def row_op[DOT: Bool](self, row: Int, dst: FPtr, x: FPtr) -> F32x16:
        """Dequantizes a row into dst or, with DOT, returns the partial sums
        of its dot product with x (16 lanes, to be added up)."""
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        var gpr = self.cols // self.group
        var acc = F32x16(0)
        for g in range(gpr):
            var s = F32x16(self.scale[unsafe_offset = row * gpr + g].cast[DType.float32]())
            var zp = F32x16(Float32(Int(self.zero_point[unsafe_offset = row * gpr + g])))
            var accg = F32x16(0)  # this group's sum of x * (u - zero_point)
            for c in range(self.group // 32):
                var k0 = g * self.group + 32 * c
                var v0: F32x16
                var v1: F32x16
                comptime if Self.BITS == 4:
                    var b = p.unsafe_load[width=16](k0 // 2)
                    v0 = (b & 0xF).cast[DType.float32]() - zp
                    v1 = (b >> 4).cast[DType.float32]() - zp
                else:
                    v0 = p.unsafe_load[width=16](k0).cast[DType.float32]() - zp
                    v1 = p.unsafe_load[width=16](k0 + 16).cast[DType.float32]() - zp
                comptime if DOT:
                    accg = v0.fma(x.unsafe_load[width=16](k0), accg)
                    accg = v1.fma(x.unsafe_load[width=16](k0 + 16), accg)
                else:
                    dst.unsafe_store(k0, v0 * s)
                    dst.unsafe_store(k0 + 16, v1 * s)
            comptime if DOT:
                acc = accg.fma(s, acc)
        return acc

    def dequant_row(self, row: Int, dst: FPtr):
        _ = self.row_op[False](row, dst, dst)

    def dot_row(self, row: Int, x: FPtr) -> Float32:
        return self.row_op[True](row, x, x).reduce_add()

    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """Slow path, for completeness: dequantizes the whole row. The
        kernels use dequant_row and dot_row instead."""
        var buf = unsafe_alloc[Float32](self.cols)
        self.dequant_row(row, buf)
        var v = buf.unsafe_load[width=width](col)
        buf.unsafe_free()
        return v

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
