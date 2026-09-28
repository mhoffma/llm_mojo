"""Reading llama.cpp GGUF files and dequantizing their block formats.

GGUF layout (version 3, little-endian):

    magic "GGUF" | version u32 | tensor count u64 | metadata count u64
    metadata: key (string), value type u32, value
    tensor infos: name (string), n_dims u32, dims u64[n_dims], type u32,
                  offset u64 (from the start of the data section)
    padding to `general.alignment` (default 32), then the tensor data

Strings are a u64 length followed by bytes. Dimension 0 is the contiguous
one, so a matrix with dims [IN, OUT] is stored as OUT rows of IN values: one
row per output, each row a run of quantization blocks along the input axis.

Block formats implemented here (from ggml's ggml-quants.c):

    type   weights/block  bytes  layout
    Q4_0        32          18   f16 d; 16 bytes of 4-bit q.   w = d*(q-8)
    Q4_1        32          20   f16 d, f16 m; 16 bytes q.     w = d*q + m
    Q8_0        32          34   f16 d; 32 int8 q.             w = d*q
    Q4_K       256         144   f16 d, dmin; 12 bytes of 6-bit sub-block
                                 scales and mins; 128 bytes q (8 sub-blocks
                                 of 32).   w = d*sc*q - dmin*m
    Q5_K       256         176   like Q4_K plus 32 bytes holding each
                                 weight's 5th bit
    Q6_K       256         210   128 bytes low 4 bits, 64 bytes high 2 bits,
                                 16 int8 sub-block scales, f16 d.
                                 w = d*sc*(q-32)
"""

from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.collections import Dict
from std.os import abort, SEEK_END, SEEK_SET
from std.utils import IndexList

from tensor import FPtr, NW, F32V, WeightMatrix, I16Ptr, I16x32, I32x16, dot_pairs

comptime BPtr = Pointer[UInt8, MutUntrackedOrigin]

# ggml tensor types (ggml.h).
comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q4_0 = 2
comptime GGML_Q4_1 = 3
comptime GGML_Q8_0 = 8
comptime GGML_Q4_K = 12
comptime GGML_Q5_K = 13
comptime GGML_Q6_K = 14


def type_name(t: Int) -> String:
    if t == GGML_F32:
        return "F32"
    if t == GGML_F16:
        return "F16"
    if t == GGML_Q4_0:
        return "Q4_0"
    if t == GGML_Q4_1:
        return "Q4_1"
    if t == GGML_Q8_0:
        return "Q8_0"
    if t == GGML_Q4_K:
        return "Q4_K"
    if t == GGML_Q5_K:
        return "Q5_K"
    if t == GGML_Q6_K:
        return "Q6_K"
    return "type " + String(t)


def block_size(t: Int) -> Int:
    """Weights per block."""
    if t == GGML_F32 or t == GGML_F16:
        return 1
    if t == GGML_Q4_0 or t == GGML_Q4_1 or t == GGML_Q8_0:
        return 32
    return 256


def block_bytes(t: Int) -> Int:
    if t == GGML_F32:
        return 4
    if t == GGML_F16:
        return 2
    if t == GGML_Q4_0:
        return 18
    if t == GGML_Q4_1:
        return 20
    if t == GGML_Q8_0:
        return 34
    if t == GGML_Q4_K:
        return 144
    if t == GGML_Q5_K:
        return 176
    if t == GGML_Q6_K:
        return 210
    return 0


# ===----------------------------------------------------------------------=== #
# Little-endian reads from a byte pointer
# ===----------------------------------------------------------------------=== #


@always_inline
def u16_at(p: BPtr, i: Int) -> Int:
    return Int(p[unsafe_offset=i]) | (Int(p[unsafe_offset = i + 1]) << 8)


@always_inline
def u32_at(p: BPtr, i: Int) -> Int:
    return u16_at(p, i) | (u16_at(p, i + 2) << 16)


@always_inline
def u64_at(p: BPtr, i: Int) -> Int:
    return u32_at(p, i) | (u32_at(p, i + 4) << 32)


@always_inline
def f16_at(p: BPtr, i: Int) -> Float32:
    """Reads a float16 and widens it to float32."""
    var bits = SIMD[DType.uint16, 1](UInt16(u16_at(p, i)))
    return bitcast[DType.float16, 1](bits).cast[DType.float32]()


@always_inline
def f32_at(p: BPtr, i: Int) -> Float32:
    var bits = SIMD[DType.uint32, 1](UInt32(u32_at(p, i)))
    return bitcast[DType.float32, 1](bits)


# ===----------------------------------------------------------------------=== #
# Dequantizing one block into dst
# ===----------------------------------------------------------------------=== #


# Each dequantizer works on 16 weights at a time: it loads 16 bytes as a
# SIMD[uint8, 16], splits nibbles or bit fields with masks and shifts (which
# act on all 16 lanes at once), converts the codes to float32 in one step,
# and applies the block's scale and offset. These replaced a first version
# that did the same per weight in scalar code, which was ~10x slower.
#
# Each one is compiled in two variants, chosen by the DOT parameter:
#   DOT=False: store the weights to dst (dequantize a row);
#   DOT=True:  multiply them by x and add into acc, without storing (the
#              dot product of a row with x, fused; used when decoding).
# `put` is the only place the variants differ, so they can't drift apart.

comptime U8x16 = SIMD[DType.uint8, 16]
comptime F32x16 = SIMD[DType.float32, 16]


@always_inline
def bytes16(p: BPtr, i: Int) -> U8x16:
    return p.unsafe_load[width=16](i)


@always_inline
def codes(v: U8x16) -> F32x16:
    """Unsigned byte codes to float32."""
    return v.cast[DType.float32]()


@always_inline
def put[
    DOT: Bool
](dst: FPtr, x: FPtr, acc: F32x16, off: Int, v: F32x16) -> F32x16:
    """Stores v to dst[off:], or with DOT, returns acc + v * x[off:]."""
    comptime if DOT:
        return v.fma(x.unsafe_load[width=16](off), acc)
    else:
        dst.unsafe_store(off, v)
        return acc


def dequant_q4_0[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = F32x16(f16_at(b, 0))
    var q = bytes16(b, 2)
    # Low nibbles are weights 0-15, high nibbles 16-31.
    acc = put[DOT](dst, x, acc, 0, (codes(q & 0xF) - 8) * d)
    return put[DOT](dst, x, acc, 16, (codes(q >> 4) - 8) * d)


def dequant_q4_1[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = F32x16(f16_at(b, 0))
    var m = F32x16(f16_at(b, 2))
    var q = bytes16(b, 4)
    acc = put[DOT](dst, x, acc, 0, d * codes(q & 0xF) + m)
    return put[DOT](dst, x, acc, 16, d * codes(q >> 4) + m)


def dequant_q8_0[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = F32x16(f16_at(b, 0))
    comptime for h in range(2):
        var q = bitcast[DType.int8, 16](bytes16(b, 2 + 16 * h))
        acc = put[DOT](dst, x, acc, 16 * h, d * q.cast[DType.float32]())
    return acc


@always_inline
def k_scale_min(q: BPtr, j: Int) -> Tuple[Int, Int]:
    """Unpacks sub-block j's 6-bit scale and min from a K-quant's 12 bytes.

    The first four sub-blocks keep them in the low 6 bits of bytes 0-7; the
    last four split them into a low nibble (bytes 8-11) and the top 2 bits
    of bytes 0-7.
    """
    if j < 4:
        return (Int(q[unsafe_offset=j]) & 63, Int(q[unsafe_offset = j + 4]) & 63)
    var sc = (Int(q[unsafe_offset = j + 4]) & 0xF) | (
        (Int(q[unsafe_offset = j - 4]) >> 6) << 4
    )
    var m = (Int(q[unsafe_offset = j + 4]) >> 4) | (
        (Int(q[unsafe_offset=j]) >> 6) << 4
    )
    return (sc, m)


def dequant_q4_k[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = f16_at(b, 0)
    var dmin = f16_at(b, 2)
    var scales = b.unsafe_offset(4)
    # 4 chunks of 64 weights. Chunk j's 32 bytes hold sub-block 2j in their
    # low nibbles and sub-block 2j+1 in their high nibbles.
    comptime for j in range(4):
        var sm1 = k_scale_min(scales, 2 * j)
        var sm2 = k_scale_min(scales, 2 * j + 1)
        var d1 = F32x16(d * Float32(sm1[0]))
        var m1 = F32x16(dmin * Float32(sm1[1]))
        var d2 = F32x16(d * Float32(sm2[0]))
        var m2 = F32x16(dmin * Float32(sm2[1]))
        comptime for h in range(2):
            var q = bytes16(b, 16 + 32 * j + 16 * h)
            acc = put[DOT](dst, x, acc, 64 * j + 16 * h, d1 * codes(q & 0xF) - m1)
            acc = put[DOT](
                dst, x, acc, 64 * j + 32 + 16 * h, d2 * codes(q >> 4) - m2
            )
    return acc


def dequant_q5_k[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = f16_at(b, 0)
    var dmin = f16_at(b, 2)
    var scales = b.unsafe_offset(4)
    # Like Q4_K, plus a 5th bit per weight: bit 2j of qh[l] for the low-nibble
    # weight l of chunk j, bit 2j+1 for the high-nibble one.
    comptime for j in range(4):
        var sm1 = k_scale_min(scales, 2 * j)
        var sm2 = k_scale_min(scales, 2 * j + 1)
        var d1 = F32x16(d * Float32(sm1[0]))
        var m1 = F32x16(dmin * Float32(sm1[1]))
        var d2 = F32x16(d * Float32(sm2[0]))
        var m2 = F32x16(dmin * Float32(sm2[1]))
        comptime for h in range(2):
            var qh = bytes16(b, 16 + 16 * h)
            var q = bytes16(b, 48 + 32 * j + 16 * h)
            var lo = (q & 0xF) | (((qh >> U8x16(2 * j)) & 1) << 4)
            var hi = (q >> 4) | (((qh >> U8x16(2 * j + 1)) & 1) << 4)
            acc = put[DOT](dst, x, acc, 64 * j + 16 * h, d1 * codes(lo) - m1)
            acc = put[DOT](dst, x, acc, 64 * j + 32 + 16 * h, d2 * codes(hi) - m2)
    return acc


def dequant_q6_k[DOT: Bool](b: BPtr, dst: FPtr, x: FPtr, acc_in: F32x16) -> F32x16:
    var acc = acc_in
    var d = f16_at(b, 208)
    # 2 halves of 128 weights. In half n, byte l of ql (64 bytes) and qh (32
    # bytes) supply weights l, l+32, l+64, l+96: low 4 bits from a nibble of
    # ql, high 2 bits from a pair of bits in qh. Each 16 weights share one
    # int8 scale.
    comptime for n in range(2):
        var ql = 64 * n
        var qhoff = 128 + 32 * n
        var sc = 192 + 8 * n
        comptime for h in range(2):
            var lo0 = bytes16(b, ql + 16 * h)
            var lo1 = bytes16(b, ql + 32 + 16 * h)
            var hi = bytes16(b, qhoff + 16 * h)
            var q1 = (lo0 & 0xF) | ((hi & 3) << 4)
            var q2 = (lo1 & 0xF) | (((hi >> 2) & 3) << 4)
            var q3 = (lo0 >> 4) | (((hi >> 4) & 3) << 4)
            var q4 = (lo1 >> 4) | (((hi >> 6) & 3) << 4)
            var base = 128 * n + 16 * h
            comptime for k in range(4):
                var s = Float32(Int(b[unsafe_offset = sc + h + 2 * k].cast[DType.int8]()))
                var qk = q1 if k == 0 else (q2 if k == 1 else (q3 if k == 2 else q4))
                acc = put[DOT](
                    dst, x, acc, base + 32 * k, F32x16(d * s) * (codes(qk) - 32)
                )
    return acc


@always_inline
def q8_0_i16(b: BPtr) -> I16x32:
    """The 32 int8 weights of the Q8_0 block at b, as int16."""
    return bitcast[DType.int8, 32](b.unsafe_load[width=32](2)).cast[DType.int16]()


@always_inline
def q6_k_scales(b: BPtr) -> I16x32:
    """The 16 int8 scales of the Q6_K superblock at b as int16, in lanes
    0-15 (and repeated in 16-31), for q6_k_i16's VPERMW."""
    var s = b.unsafe_load[width=16](192).cast[DType.int8]().cast[DType.int16]()
    return s.join(s)


def scale_mask(g: Int) -> IndexList[32]:
    """Shuffle mask: lanes 0-15 take scale g, lanes 16-31 scale g + 1."""
    var m = IndexList[32]()
    for i in range(32):
        m[i] = g if i < 16 else g + 1
    return m


@always_inline
def q6_k_i16[n: Int, k: Int](b: BPtr, sv: I16x32) -> I16x32:
    """Weights 128n + 32k .. +31 of the Q6_K superblock at b as int16
    (q - 32) * scale. The same bytes as dequant_q6_k, read 32 at a time:
    the 32 weights are half n's h = 0 and h = 1 vectors for this k, whose
    16-weight groups have scales 8n + 2k and 8n + 2k + 1.

    The scale vector is a shuffle of sv (q6_k_scales): ~12% faster than
    building it from two scalar loads (research/test_head.mojo)."""
    var L = b.unsafe_load[width=32](64 * n + (32 if k % 2 == 1 else 0))
    var H = b.unsafe_load[width=32](128 + 32 * n)
    comptime SHIFT = SIMD[DType.uint8, 32](2 * k)  # this k's pair of high bits
    var q = (((L >> 4) if k >= 2 else (L & 0xF)) | (((H >> SHIFT) & 3) << 4))
    comptime MASK = scale_mask(8 * n + 2 * k)
    return (q.cast[DType.int16]() - 32) * sv.shuffle[MASK]()


# ===----------------------------------------------------------------------=== #
# A GGUF tensor as a weight matrix
# ===----------------------------------------------------------------------=== #


struct GGUFMatrix(WeightMatrix):
    """One GGUF tensor: `rows` rows of `cols` values, in its file format.

    The data stays in the file's buffer (owned by whoever read the file), so
    `free` does nothing. Rows are [OUT, IN] for layer matrices (OUT_MAJOR),
    and [V, C] for the embedding and output head, the same as wte.

    The format is a runtime property (`kind`), because a GGUF file mixes
    formats: in Q4_K_M, one layer's MLP matrix can be Q4_K and the next Q6_K.
    The branch on it is taken once per row.
    """

    comptime OUT_MAJOR = True
    comptime FROM_GGUF = True
    comptime ACT16 = False

    var data: BPtr
    var kind: Int
    var rows: Int
    var cols: Int
    var row_bytes: Int

    def __init__(out self, data: BPtr, kind: Int, rows: Int, cols: Int):
        self.data = data
        self.kind = kind
        self.rows = rows
        self.cols = cols
        self.row_bytes = cols // block_size(kind) * block_bytes(kind)

    @staticmethod
    def name() -> String:
        return "gguf"

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self:
        abort("GGUFMatrix is loaded from a .gguf file, not converted")

    def dequant_row(self, row: Int, dst: FPtr):
        _ = self.each_block[False](row, dst, dst)

    def dot_row(self, row: Int, x: FPtr) -> Float32:
        """Dot product of row `row` with x[cols], dequantizing on the fly."""
        return self.each_block[True](row, x, x).reduce_add()

    @always_inline
    def each_block[DOT: Bool](self, row: Int, dst: FPtr, x: FPtr) -> F32x16:
        """Runs the row's format's dequantizer over each of its blocks."""
        var p = self.data.unsafe_offset(row * self.row_bytes)
        var k = self.kind
        # One branch per row, then a tight loop over the row's blocks.
        if k == GGML_Q4_0:
            return self.blocks[dequant_q4_0[DOT]](p, dst, x)
        if k == GGML_Q4_1:
            return self.blocks[dequant_q4_1[DOT]](p, dst, x)
        if k == GGML_Q8_0:
            return self.blocks[dequant_q8_0[DOT]](p, dst, x)
        if k == GGML_Q4_K:
            return self.blocks[dequant_q4_k[DOT]](p, dst, x)
        if k == GGML_Q5_K:
            return self.blocks[dequant_q5_k[DOT]](p, dst, x)
        if k == GGML_Q6_K:
            return self.blocks[dequant_q6_k[DOT]](p, dst, x)
        # F32 and F16: plain loops.
        var acc = F32x16(0)
        for i in range(0, self.cols, 16):
            var v: F32x16
            if k == GGML_F32:
                v = p.unsafe_bitcast[Float32]().unsafe_load[width=16, alignment=1](i)
            else:
                v = p.unsafe_bitcast[Float16]().unsafe_load[width=16, alignment=1](
                    i
                ).cast[DType.float32]()
            acc = put[DOT](dst, x, acc, i, v)
        return acc

    @always_inline
    def blocks[
        f: def(BPtr, FPtr, FPtr, F32x16) thin -> F32x16
    ](self, p: BPtr, dst: FPtr, x: FPtr) -> F32x16:
        """Runs f on every block of the row starting at p."""
        var bs = block_size(self.kind)
        var bb = block_bytes(self.kind)
        var acc = F32x16(0)
        for blk in range(self.cols // bs):
            acc = f(
                p.unsafe_offset(blk * bb),
                dst.unsafe_offset(blk * bs),
                x.unsafe_offset(blk * bs),
                acc,
            )
        return acc

    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """Slow path, for completeness: dequantizes the whole row. The
        kernels use dequant_row for GGUF matrices instead."""
        var buf = unsafe_alloc[Float32](self.cols)
        self.dequant_row(row, buf)
        var v = buf.unsafe_load[width=width](col)
        buf.unsafe_free()
        return v

    # Integer path, for the formats of the output heads of GPT-2's GGUF
    # files (Q6_K in Q4_0 and Q4_K_M, Q8_0 in Q8_0): activations as int16
    # (kernels.quantize_rows), weights as int16, products summed in int32
    # with VPDPWSSD, then each group's sum times its float scale d:
    # - Q8_0: groups of 32, weights q. Each int32 lane sums 2 products.
    # - Q6_K: groups of 256 (a superblock), weights (q - 32) * scale with the
    #   16-weight group's int8 scale. |(q - 32) * scale| <= 32 * 128 = 4096,
    #   so each int32 lane (16 products) stays within
    #   16 * 4096 * 32767 = 2,147,418,112 < 2^31.
    # research/test_head.mojo measured the speed and error.

    def has_i16(self) -> Bool:
        return self.kind == GGML_Q6_K or self.kind == GGML_Q8_0

    def group_size(self) -> Int:
        return 256 if self.kind == GGML_Q6_K else 32

    def dot_row_i16(self, row: Int, xq: I16Ptr, xsums: FPtr) -> Float32:
        """Unpacks and multiplies in one pass, in the same order as
        kernels.dot1_i16 on unpack_row_i16's output (identical results)."""
        var p = self.data.unsafe_offset(row * self.row_bytes)
        var accf = F32x16(0)
        if self.kind == GGML_Q8_0:
            for blk in range(self.cols // 32):
                var b = p.unsafe_offset(blk * 34)
                var acc = dot_pairs(
                    I32x16(0), xq.unsafe_load[width=32](blk * 32), q8_0_i16(b)
                )
                accf = acc.cast[DType.float32]().fma(F32x16(f16_at(b, 0)), accf)
            return accf.reduce_add()
        for sb in range(self.cols // 256):
            var b = p.unsafe_offset(sb * 210)
            var sv = q6_k_scales(b)
            var acc = I32x16(0)
            comptime for n in range(2):
                comptime for k in range(4):
                    acc = dot_pairs(
                        acc,
                        xq.unsafe_load[width=32](sb * 256 + 128 * n + 32 * k),
                        q6_k_i16[n, k](b, sv),
                    )
            accf = acc.cast[DType.float32]().fma(F32x16(f16_at(b, 208)), accf)
        return accf.reduce_add()

    def unpack_row_i16(self, row: Int, dst: I16Ptr, scales: FPtr):
        var p = self.data.unsafe_offset(row * self.row_bytes)
        if self.kind == GGML_Q8_0:
            for blk in range(self.cols // 32):
                var b = p.unsafe_offset(blk * 34)
                dst.unsafe_store(blk * 32, q8_0_i16(b))
                scales[unsafe_offset=blk] = f16_at(b, 0)
            return
        for sb in range(self.cols // 256):
            var b = p.unsafe_offset(sb * 210)
            var sv = q6_k_scales(b)
            comptime for n in range(2):
                comptime for k in range(4):
                    dst.unsafe_store(
                        sb * 256 + 128 * n + 32 * k, q6_k_i16[n, k](b, sv)
                    )
            scales[unsafe_offset=sb] = f16_at(b, 208)

    def nbytes(self) -> Int:
        return self.rows * self.row_bytes

    def free(self):
        pass


# ===----------------------------------------------------------------------=== #
# The file
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var kind: Int
    var cols: Int  # dims[0], the contiguous dimension
    var rows: Int  # product of the remaining dims
    var offset: Int  # from the start of the data section


struct GGUFFile(Movable):
    """A GGUF file read into memory, with its tensor index."""

    var buf: BPtr
    var size: Int
    var data_start: Int
    var tensors: Dict[String, TensorInfo]

    def __init__(out self, path: String) raises:
        var f = open(path, "r")
        self.size = Int(f.seek(0, SEEK_END))
        _ = f.seek(0, SEEK_SET)
        self.buf = unsafe_alloc[UInt8](self.size)
        var done = 0
        while done < self.size:
            var n = f.read(
                Span(unsafe_ptr=self.buf.unsafe_offset(done), length=self.size - done)
            )
            if n <= 0:
                raise Error("short read on " + path)
            done += n
        f.close()
        self.tensors = Dict[String, TensorInfo]()
        self.data_start = 0

        var p = self.buf
        if u32_at(p, 0) != 0x46554747:  # "GGUF"
            raise Error(path + " is not a GGUF file")
        var version = u32_at(p, 4)
        if version < 2:
            raise Error("GGUF version " + String(version) + " is not supported")
        var n_tensors = u64_at(p, 8)
        var n_kv = u64_at(p, 16)
        var pos = 24
        var alignment = 32
        for _ in range(n_kv):
            var key = self.string_at(pos)
            pos += 8 + key.byte_length()
            var vtype = u32_at(p, pos)
            pos += 4
            if key == "general.alignment":
                alignment = u32_at(p, pos)
            pos = self.skip_value(vtype, pos)
        for _ in range(n_tensors):
            var name = self.string_at(pos)
            pos += 8 + name.byte_length()
            var ndims = u32_at(p, pos)
            pos += 4
            var cols = u64_at(p, pos)
            var rows = 1
            for d in range(1, ndims):
                rows *= u64_at(p, pos + 8 * d)
            pos += 8 * ndims
            var kind = u32_at(p, pos)
            var offset = u64_at(p, pos + 4)
            pos += 12
            self.tensors[name] = TensorInfo(kind, cols, rows, offset)
        self.data_start = (pos + alignment - 1) // alignment * alignment

    def __deinit__(deinit self):
        self.buf.unsafe_free()

    def release(deinit self) -> BPtr:
        """Consumes the file and hands its buffer (and the duty to free it)
        to the caller. Matrices from `matrix` point into this buffer."""
        return self.buf

    def string_at(self, pos: Int) -> String:
        var n = u64_at(self.buf, pos)
        return String(
            from_utf8_lossy=Span(unsafe_ptr=self.buf.unsafe_offset(pos + 8), length=n)
        )

    def skip_value(self, vtype: Int, pos: Int) -> Int:
        """Returns the position just past a metadata value of type vtype."""
        if vtype == 0 or vtype == 1 or vtype == 7:  # u8, i8, bool
            return pos + 1
        if vtype == 2 or vtype == 3:  # u16, i16
            return pos + 2
        if vtype == 4 or vtype == 5 or vtype == 6:  # u32, i32, f32
            return pos + 4
        if vtype == 10 or vtype == 11 or vtype == 12:  # u64, i64, f64
            return pos + 8
        if vtype == 8:  # string
            return pos + 8 + u64_at(self.buf, pos)
        if vtype == 9:  # array: element type, count, elements
            var etype = u32_at(self.buf, pos)
            var count = u64_at(self.buf, pos + 4)
            var q = pos + 12
            for _ in range(count):
                q = self.skip_value(etype, q)
            return q
        abort("unknown GGUF metadata type " + String(vtype))

    def matrix(self, name: String) raises -> GGUFMatrix:
        var info = self.tensors.get(name)
        if not info:
            raise Error("tensor not in GGUF file: " + name)
        var t = info.value().copy()
        if block_bytes(t.kind) == 0:
            raise Error(name + ": unsupported format " + type_name(t.kind))
        return GGUFMatrix(
            self.buf.unsafe_offset(self.data_start + t.offset),
            t.kind,
            t.rows,
            t.cols,
        )

    def has(self, name: String) -> Bool:
        return name in self.tensors
