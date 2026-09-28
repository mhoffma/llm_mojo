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

from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic, prefetch
from std.sys import simd_width_of, size_of
from std.os import abort
from std.math import round

comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime NW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, NW]
comptime F32x16 = SIMD[DType.float32, 16]
comptime I16Ptr = Pointer[Int16, MutUntrackedOrigin]
comptime I16x32 = SIMD[DType.int16, 32]
comptime I32x16 = SIMD[DType.int32, 16]


@always_inline
def dot_pairs(acc: I32x16, a: I16x32, b: I16x32) -> I32x16:
    """acc[i] + a[2i]*b[2i] + a[2i+1]*b[2i+1]: int16 products summed into
    int32. One VPDPWSSD instruction with AVX-512 VNNI (see
    research/test_vnni.mojo); otherwise portable, much slower SIMD code."""
    comptime if CompilationTarget.has_vnni():
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.512", I32x16, has_side_effect=False
        ](acc, a, b)
    else:
        var p = a.cast[DType.int32]() * b.cast[DType.int32]()
        var halves = p.deinterleave()
        return acc + halves[0] + halves[1]


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

    comptime ACT16: Bool
    """True if matmuls with this format quantize their input activations to
    int16 and compute in integers (kernels.matmul_rows_a16), using the
    *_i16 methods below. Only QuantMatrix[..., A16=True] does."""

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

    def has_i16(self) -> Bool:
        """Whether this matrix has the integer methods below (dot_row_i16,
        unpack_row_i16, group_size): always for ACT16 formats; for formats
        that mix block types at runtime (GGUF), per matrix."""
        return Self.ACT16

    def permute_x_i16(self, xq: I16Ptr, n: Int, dst: I16Ptr, sums: FPtr) -> Bool:
        """For ACT16 formats whose decode layout needs it: writes the int16
        activations xq[n] into dst in the order dot_row_i16 reads them, and
        per-lane activation sums into sums[n / 16], once per token. Returns
        False (and writes nothing) if the format reads xq as is."""
        return False

    def dot_row_i16(self, row: Int, xq: I16Ptr, xsums: FPtr) -> Float32:
        """For ACT16 formats: the dot product of row `row` with int16
        activations xq[cols], in the weights' float scale (the caller
        multiplies by the activations' scale). xq and xsums are as prepared
        by permute_x_i16 if it returned True."""
        abort("dot_row_i16: this format has no integer kernel")

    def unpack_row_i16(self, row: Int, dst: I16Ptr, scales: FPtr):
        """For ACT16 formats: writes row `row` as int16 codes minus the zero
        point (dst[cols]) and its group scales (scales[cols / group])."""
        abort("unpack_row_i16: this format has no integer kernel")

    def group_size(self) -> Int:
        """For ACT16 formats: weights per quantization group."""
        return 0

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
    comptime ACT16 = False

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


struct QuantMatrix[
    BITS: Int, GROUP: Int, SYMMETRIC: Bool, A16: Bool = False
](WeightMatrix):
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
    scale once per group.

    With A16 (W4A16 / W8A16), matmuls quantize their inputs to int16 and the
    inner sums run in integers: `dot_row_i16` unpacks 32 codes to int16,
    subtracts the zero point, and multiplies-and-adds them with 32 int16
    activations in one VPDPWSSD; each group's int32 sums then take the
    group's scale in float. |x_q| <= 32767 and |u - zero_point| <= 255, and
    here each of the 16 int32 lanes sums 2 of every 32 products, so a lane
    gathers at most 2 * 32767 * 255 per 32 weights and even a 3072-wide group
    (192 products per lane) stays below 2^31. (The prefill tile kernel sums
    a whole group in one lane, so it flushes to float every 256 inputs; see
    kernels.tile_i16.)

    Decode layout for int4, 32-weight groups, A16 (BLOCKED): in the layout
    above each group fills a whole register, so every 32 weights need their
    own int32 -> float conversion and scale. Instead, each 256 weights (8
    groups) are stored as 8 steps of 16 bytes, where step s's lane i (int16
    pair i of the unpacked 32 codes) holds codes 2p and 2p+1 of group i % 8,
    with p = 8 * (i // 8) + s: lanes 0-7 cover the first halves of the 8
    groups and lanes 8-15 the second halves. After 8 VPDPWSSDs each lane
    holds one half-group's sum, and zero points, conversion and scales are
    applied once per 256 weights, as 16-lane vector operations. The zero
    point is applied as sum(x*u) - zero_point*sum(x), with the per-lane
    activation sums computed once per token (permute_x_i16, which also puts
    the activations in the same order). Only int4-g32*-a16 use this layout.

    Rounding is plain round-to-nearest.
    """

    comptime OUT_MAJOR = True
    comptime FROM_GGUF = False
    comptime ACT16 = Self.A16
    comptime BLOCKED = Self.A16 and Self.BITS == 4 and Self.GROUP == 32
    """Group-per-lane code layout, for fast decode with 32-weight groups
    (see the struct docstring)."""
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
        if Self.SYMMETRIC:
            s += "-sym"
        return s + "-a16" if Self.A16 else s

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

    @always_inline
    @staticmethod
    def nibble_at(col: Int) -> Tuple[Int, Bool]:
        """For int4: the byte (within its row) holding column col's code,
        and whether it is the high nibble. Unpacking 16 bytes b as
        (b & 0xF).join(b >> 4) gives 32 codes where code j comes from byte
        j % 16, high nibble for j >= 16."""
        var j: Int  # position among the 32 codes unpacked together
        var chunk: Int  # which 16 bytes
        comptime if Self.BLOCKED:
            var w = col % 256
            var e = w % 32  # position in its group
            var p = e // 2  # pair in the group
            var lane = w // 32 + 8 * (p // 8)
            chunk = (col // 256) * 8 + p % 8  # block, then step
            j = 2 * lane + e % 2
        else:
            chunk = col // 32
            j = col % 32
        return (chunk * 16 + j % 16, j >= 16)

    def set_code(self, row: Int, col: Int, u: Int):
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        comptime if Self.BITS == 8:
            p[unsafe_offset=col] = UInt8(u)
        else:
            var at = Self.nibble_at(col)
            var b = p[unsafe_offset = at[0]]
            if at[1]:
                b = (b & 0x0F) | UInt8(u << 4)
            else:
                b = (b & 0xF0) | UInt8(u)
            p[unsafe_offset = at[0]] = b

    @always_inline
    def code(self, row: Int, col: Int) -> Int:
        """One code, read with nibble_at (scalar; for the BLOCKED layout's
        slow paths)."""
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        comptime if Self.BITS == 8:
            return Int(p[unsafe_offset=col])
        else:
            var at = Self.nibble_at(col)
            var b = Int(p[unsafe_offset = at[0]])
            return b >> 4 if at[1] else b & 0xF

    @always_inline
    def row_op[DOT: Bool](self, row: Int, dst: FPtr, x: FPtr) -> F32x16:
        """Dequantizes a row into dst or, with DOT, returns the partial sums
        of its dot product with x (16 lanes, to be added up)."""
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        var gpr = self.cols // self.group
        var acc = F32x16(0)
        comptime if Self.BLOCKED:
            # Scalar path: only the embedding lookup and tests use this
            # layout's float methods.
            var d = Float32(0)
            for k in range(self.cols):
                var g = row * gpr + k // self.group
                var v = Float32(
                    self.code(row, k) - Int(self.zero_point[unsafe_offset=g])
                ) * self.scale[unsafe_offset=g].cast[DType.float32]()
                comptime if DOT:
                    d += v * x[unsafe_offset=k]
                else:
                    dst[unsafe_offset=k] = v
            acc[0] = d
            return acc
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

    @always_inline
    def codes_i16(self, p: Pointer[UInt8, MutUntrackedOrigin], k0: Int, zp: I16x32) -> I16x32:
        """32 codes starting at column k0 of the row at p, as int16, minus
        the zero point. With the Q4_0-style packing, the low nibbles of 16
        bytes are weights k0..k0+15 and the high nibbles k0+16..k0+31, so
        joining them gives the 32 weights in order."""
        var u: SIMD[DType.uint8, 32]
        comptime if Self.BITS == 4:
            var b = p.unsafe_load[width=16](k0 // 2)
            u = (b & 0xF).join(b >> 4)
        else:
            u = p.unsafe_load[width=32](k0)
        return u.cast[DType.int16]() - zp

    def permute_x_i16(self, xq: I16Ptr, n: Int, dst: I16Ptr, sums: FPtr) -> Bool:
        comptime if Self.BLOCKED:
            # Put activation k where the code of column k sits after
            # unpacking, and sum each lane's 16 activations (exact in float:
            # at most 16 * 32767).
            for blk in range(n // 256):
                for lane in range(16):
                    sums[unsafe_offset = blk * 16 + lane] = 0
                for w in range(256):
                    var col = blk * 256 + w
                    var e = w % 32
                    var p = e // 2
                    var lane = w // 32 + 8 * (p // 8)
                    var v = xq[unsafe_offset=col]
                    dst[unsafe_offset = blk * 256 + (p % 8) * 32 + 2 * lane + e % 2] = v
                    sums[unsafe_offset = blk * 16 + lane] += Float32(Int(v))
            return True
        else:
            return False

    def dot_row_i16(self, row: Int, xq: I16Ptr, xsums: FPtr) -> Float32:
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        var gpr = self.cols // self.group
        var accf = F32x16(0)
        comptime if Self.BLOCKED:
            for blk in range(self.cols // 256):
                var acc = I32x16(0)
                comptime for step in range(8):
                    var b = p.unsafe_load[width=16](blk * 128 + step * 16)
                    var u = (b & 0xF).join(b >> 4).cast[DType.int16]()
                    acc = dot_pairs(
                        acc, xq.unsafe_load[width=32](blk * 256 + step * 32), u
                    )
                # Lane i belongs to group i % 8: duplicate the 8 groups' zero
                # points and scales across both halves.
                var g0 = row * gpr + blk * 8
                var zp8 = self.zero_point.unsafe_load[width=8](g0)
                var sc8 = self.scale.unsafe_load[width=8](g0)
                var zp = zp8.join(zp8).cast[DType.float32]()
                var sc = sc8.join(sc8).cast[DType.float32]()
                var xs = xsums.unsafe_load[width=16](blk * 16)
                accf = (acc.cast[DType.float32]() - zp * xs).fma(sc, accf)
            return accf.reduce_add()
        comptime if Self.GROUP == 0 and Self.BITS == 8:
            # Per-channel int8 (the output head, int8-ch layers): one group
            # per row. The loop over the row is unrolled 4x, and the row 8
            # ahead is prefetched (a prefetch never faults, even past the
            # end); together ~15% faster on the memory-bound output head
            # (research/test_head.mojo). Same sums in the same order as the
            # general loop below, so the result is identical.
            var ahead = p.unsafe_offset(8 * self.cols)
            for l in range(0, self.cols, 64):
                prefetch(ahead.unsafe_offset(l))
            var zp = I16x32(Int16(Int(self.zero_point[unsafe_offset=row])))
            var acc = I32x16(0)
            var k = 0
            while k + 128 <= self.cols:
                comptime for u in range(4):
                    acc = dot_pairs(
                        acc,
                        xq.unsafe_load[width=32](k + 32 * u),
                        self.codes_i16(p, k + 32 * u, zp),
                    )
                k += 128
            while k < self.cols:
                acc = dot_pairs(acc, xq.unsafe_load[width=32](k), self.codes_i16(p, k, zp))
                k += 32
            var s = self.scale[unsafe_offset=row].cast[DType.float32]()
            return acc.cast[DType.float32]().fma(F32x16(s), F32x16(0)).reduce_add()
        for g in range(gpr):
            var zp = I16x32(Int16(Int(self.zero_point[unsafe_offset = row * gpr + g])))
            var acc = I32x16(0)
            for c in range(self.group // 32):
                var k0 = g * self.group + 32 * c
                acc = dot_pairs(
                    acc, xq.unsafe_load[width=32](k0), self.codes_i16(p, k0, zp)
                )
            var s = self.scale[unsafe_offset = row * gpr + g].cast[DType.float32]()
            accf = acc.cast[DType.float32]().fma(F32x16(s), accf)
        return accf.reduce_add()

    def unpack_row_i16(self, row: Int, dst: I16Ptr, scales: FPtr):
        var p = self.data.unsafe_offset(row * self.cols * Self.BITS // 8)
        var gpr = self.cols // self.group
        comptime if Self.BLOCKED:
            # Unpack each step's 32 codes with SIMD, subtract the lanes' zero
            # points, then store each lane's pair of codes (one int32) at its
            # place in natural order: lane L of step s holds codes 2p, 2p+1
            # of group L % 8, with p = 8 * (L // 8) + s.
            for g in range(gpr):
                scales[unsafe_offset=g] = self.scale[
                    unsafe_offset = row * gpr + g
                ].cast[DType.float32]()
            var dst32 = dst.unsafe_bitcast[Int32]()
            for blk in range(self.cols // 256):
                var zp8 = self.zero_point.unsafe_load[width=8](row * gpr + blk * 8)
                var zp32 = zp8.join(zp8).cast[DType.int32]()
                var zp = bitcast[DType.int16, 32](zp32 | (zp32 << 16))
                comptime for step in range(8):
                    var b = p.unsafe_load[width=16](blk * 128 + step * 16)
                    var u = (b & 0xF).join(b >> 4).cast[DType.int16]() - zp
                    var pairs = bitcast[DType.int32, 16](u)
                    comptime for L in range(16):
                        dst32[
                            unsafe_offset = blk * 128
                            + (L % 8) * 16
                            + (L // 8) * 8
                            + step
                        ] = pairs[L]
            return
        for g in range(gpr):
            var zp = I16x32(Int16(Int(self.zero_point[unsafe_offset = row * gpr + g])))
            scales[unsafe_offset=g] = self.scale[unsafe_offset = row * gpr + g].cast[
                DType.float32
            ]()
            for c in range(self.group // 32):
                var k0 = g * self.group + 32 * c
                dst.unsafe_store(k0, self.codes_i16(p, k0, zp))

    def group_size(self) -> Int:
        return self.group

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
