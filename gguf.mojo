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

from tensor import FPtr, NW, F32V, WeightMatrix

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


def dequant_q4_0(b: BPtr, dst: FPtr):
    var d = f16_at(b, 0)
    for j in range(16):
        var q = Int(b[unsafe_offset = 2 + j])
        dst[unsafe_offset=j] = d * Float32((q & 0xF) - 8)
        dst[unsafe_offset = j + 16] = d * Float32((q >> 4) - 8)


def dequant_q4_1(b: BPtr, dst: FPtr):
    var d = f16_at(b, 0)
    var m = f16_at(b, 2)
    for j in range(16):
        var q = Int(b[unsafe_offset = 4 + j])
        dst[unsafe_offset=j] = d * Float32(q & 0xF) + m
        dst[unsafe_offset = j + 16] = d * Float32(q >> 4) + m


def dequant_q8_0(b: BPtr, dst: FPtr):
    var d = f16_at(b, 0)
    for j in range(32):
        var q = Int(b[unsafe_offset = 2 + j].cast[DType.int8]())
        dst[unsafe_offset=j] = d * Float32(q)


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


def dequant_q4_k(b: BPtr, dst: FPtr):
    var d = f16_at(b, 0)
    var dmin = f16_at(b, 2)
    var scales = b.unsafe_offset(4)
    var q = b.unsafe_offset(16)
    var y = 0
    var sub = 0
    for _ in range(4):  # 4 x 64 weights; each byte holds two sub-blocks
        var sm1 = k_scale_min(scales, sub)
        var sm2 = k_scale_min(scales, sub + 1)
        var d1 = d * Float32(sm1[0])
        var m1 = dmin * Float32(sm1[1])
        var d2 = d * Float32(sm2[0])
        var m2 = dmin * Float32(sm2[1])
        for l in range(32):
            dst[unsafe_offset = y + l] = d1 * Float32(
                Int(q[unsafe_offset=l]) & 0xF
            ) - m1
        for l in range(32):
            dst[unsafe_offset = y + 32 + l] = d2 * Float32(
                Int(q[unsafe_offset=l]) >> 4
            ) - m2
        q = q.unsafe_offset(32)
        y += 64
        sub += 2


def dequant_q5_k(b: BPtr, dst: FPtr):
    var d = f16_at(b, 0)
    var dmin = f16_at(b, 2)
    var scales = b.unsafe_offset(4)
    var qh = b.unsafe_offset(16)
    var ql = b.unsafe_offset(48)
    var y = 0
    var sub = 0
    var u1 = 1
    var u2 = 2
    for _ in range(4):
        var sm1 = k_scale_min(scales, sub)
        var sm2 = k_scale_min(scales, sub + 1)
        var d1 = d * Float32(sm1[0])
        var m1 = dmin * Float32(sm1[1])
        var d2 = d * Float32(sm2[0])
        var m2 = dmin * Float32(sm2[1])
        for l in range(32):
            var h = 16 if (Int(qh[unsafe_offset=l]) & u1) != 0 else 0
            dst[unsafe_offset = y + l] = d1 * Float32(
                (Int(ql[unsafe_offset=l]) & 0xF) + h
            ) - m1
        for l in range(32):
            var h = 16 if (Int(qh[unsafe_offset=l]) & u2) != 0 else 0
            dst[unsafe_offset = y + 32 + l] = d2 * Float32(
                (Int(ql[unsafe_offset=l]) >> 4) + h
            ) - m2
        ql = ql.unsafe_offset(32)
        y += 64
        sub += 2
        u1 <<= 2
        u2 <<= 2


def dequant_q6_k(b: BPtr, dst: FPtr):
    var ql = b
    var qh = b.unsafe_offset(128)
    var sc = b.unsafe_offset(192)
    var d = f16_at(b, 208)
    var y = 0
    for _ in range(2):  # 2 x 128 weights
        for l in range(32):
            var s = l // 16
            var lo0 = Int(ql[unsafe_offset=l])
            var lo1 = Int(ql[unsafe_offset = l + 32])
            var hi = Int(qh[unsafe_offset=l])
            var q1 = ((lo0 & 0xF) | ((hi & 3) << 4)) - 32
            var q2 = ((lo1 & 0xF) | (((hi >> 2) & 3) << 4)) - 32
            var q3 = ((lo0 >> 4) | (((hi >> 4) & 3) << 4)) - 32
            var q4 = ((lo1 >> 4) | (((hi >> 6) & 3) << 4)) - 32
            dst[unsafe_offset = y + l] = d * Float32(
                Int(sc[unsafe_offset=s].cast[DType.int8]()) * q1
            )
            dst[unsafe_offset = y + l + 32] = d * Float32(
                Int(sc[unsafe_offset = s + 2].cast[DType.int8]()) * q2
            )
            dst[unsafe_offset = y + l + 64] = d * Float32(
                Int(sc[unsafe_offset = s + 4].cast[DType.int8]()) * q3
            )
            dst[unsafe_offset = y + l + 96] = d * Float32(
                Int(sc[unsafe_offset = s + 6].cast[DType.int8]()) * q4
            )
        ql = ql.unsafe_offset(64)
        qh = qh.unsafe_offset(32)
        sc = sc.unsafe_offset(8)
        y += 128


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
        var p = self.data.unsafe_offset(row * self.row_bytes)
        var k = self.kind
        if k == GGML_F32:
            for i in range(self.cols):
                dst[unsafe_offset=i] = f32_at(p, 4 * i)
            return
        if k == GGML_F16:
            for i in range(self.cols):
                dst[unsafe_offset=i] = f16_at(p, 2 * i)
            return
        var bs = block_size(k)
        var bb = block_bytes(k)
        for blk in range(self.cols // bs):
            var src = p.unsafe_offset(blk * bb)
            var out = dst.unsafe_offset(blk * bs)
            if k == GGML_Q4_0:
                dequant_q4_0(src, out)
            elif k == GGML_Q4_1:
                dequant_q4_1(src, out)
            elif k == GGML_Q8_0:
                dequant_q8_0(src, out)
            elif k == GGML_Q4_K:
                dequant_q4_k(src, out)
            elif k == GGML_Q5_K:
                dequant_q5_k(src, out)
            else:
                dequant_q6_k(src, out)

    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """Slow path, for completeness: dequantizes the whole row. The
        kernels use dequant_row for GGUF matrices instead."""
        var buf = unsafe_alloc[Float32](self.cols)
        self.dequant_row(row, buf)
        var v = buf.unsafe_load[width=width](col)
        buf.unsafe_free()
        return v

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
