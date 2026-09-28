"""KV cache storage formats.

The KV cache holds, for every layer, each past token's key and value vectors:
[layer, position, head, head_dim]. Attention compares the new token's query
with every cached key (`score`) and adds up the cached values weighted by the
softmax of those scores (`add_value`). How keys and values are stored is a
format, like the weights' (tensor.WeightMatrix): the attention kernel is
generic over the `KVCache` trait and each format's methods are inlined into
it.

Formats (PLAN.md, milestone M5):
- DenseKV[dtype]: float32 (the original cache), float16, bfloat16.
- QuantKV[BITS]: int16 / int8 with a scale per (layer, position, head).
"""

from std.math import round
from std.memory.alloc import unsafe_alloc
from std.sys import size_of

from tensor import FPtr, NW, F32V


trait KVCache(Deinitable, ImplicitlyCopyable):
    """Keys and values of past tokens, for every layer.

    Values are handles, like WeightMatrix: copying one copies the pointers,
    and the owner (Model) calls `free` once.
    """

    @staticmethod
    def name() -> String:
        """Short name for the format, as used by --kv."""
        ...

    @staticmethod
    def create(n_layer: Int, max_t: Int, n_head: Int, head_dim: Int) -> Self:
        """Allocates a cache for max_t positions."""
        ...

    def store(self, layer: Int, pos: Int, k: FPtr, v: FPtr):
        """Stores one token's key and value rows (n_head * head_dim float32
        values each) at `pos` of `layer`."""
        ...

    def score[HS: Int](self, layer: Int, pos: Int, head: Int, q: FPtr) -> Float32:
        """Returns q · k for the key at (layer, pos, head); q has HS values.
        """
        ...

    def add_value[
        HS: Int
    ](
        self,
        layer: Int,
        pos: Int,
        head: Int,
        p: F32V,
        mut acc: Array[F32V, length = HS // NW],
    ):
        """acc += p * v for the value at (layer, pos, head)."""
        ...

    def nbytes(self) -> Int:
        """Bytes of storage, to report."""
        ...

    def free(self):
        """Releases the storage."""
        ...


struct DenseKV[dtype: DType](KVCache):
    """Keys and values stored as `dtype`, widened to float32 when read.

    With float32 this is the original cache, doing exactly the same
    arithmetic (so the model stays bit-identical to gpt2.mojo).
    """

    var k: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var v: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var n_layer: Int
    var max_t: Int
    var dim: Int  # n_head * head_dim: one position's values

    def __init__(
        out self,
        k: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        v: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        n_layer: Int,
        max_t: Int,
        dim: Int,
    ):
        self.k = k
        self.v = v
        self.n_layer = n_layer
        self.max_t = max_t
        self.dim = dim

    @staticmethod
    def name() -> String:
        if Self.dtype == DType.float32:
            return "f32"
        if Self.dtype == DType.float16:
            return "f16"
        return "bf16"

    @staticmethod
    def create(n_layer: Int, max_t: Int, n_head: Int, head_dim: Int) -> Self:
        var n = n_layer * max_t * n_head * head_dim
        return Self(
            unsafe_alloc[Scalar[Self.dtype]](n),
            unsafe_alloc[Scalar[Self.dtype]](n),
            n_layer,
            max_t,
            n_head * head_dim,
        )

    @always_inline
    def at(self, layer: Int, pos: Int, head: Int, head_dim: Int) -> Int:
        """Offset of (layer, pos, head) in k and v."""
        return (layer * self.max_t + pos) * self.dim + head * head_dim

    def store(self, layer: Int, pos: Int, k: FPtr, v: FPtr):
        var o = self.at(layer, pos, 0, 0)
        for i in range(0, self.dim, NW):
            self.k.unsafe_store(o + i, k.unsafe_load[width=NW](i).cast[Self.dtype]())
            self.v.unsafe_store(o + i, v.unsafe_load[width=NW](i).cast[Self.dtype]())

    @always_inline
    def score[HS: Int](self, layer: Int, pos: Int, head: Int, q: FPtr) -> Float32:
        var kp = self.k.unsafe_offset(self.at(layer, pos, head, HS))
        var d = F32V(0)
        comptime for i in range(0, HS, NW):
            d += q.unsafe_load[width=NW](i) * kp.unsafe_load[width=NW](i).cast[
                DType.float32
            ]()
        return d.reduce_add()

    @always_inline
    def add_value[
        HS: Int
    ](
        self,
        layer: Int,
        pos: Int,
        head: Int,
        p: F32V,
        mut acc: Array[F32V, length = HS // NW],
    ):
        var vp = self.v.unsafe_offset(self.at(layer, pos, head, HS))
        comptime for i in range(HS // NW):
            acc[i] = p.fma(vp.unsafe_load[width=NW](i * NW).cast[DType.float32](), acc[i])

    def nbytes(self) -> Int:
        return 2 * self.n_layer * self.max_t * self.dim * size_of[Scalar[Self.dtype]]()

    def free(self):
        self.k.unsafe_free()
        self.v.unsafe_free()


struct QuantKV[BITS: Int](KVCache):
    """Keys and values as BITS-bit signed integers (16 or 8), with one float32
    scale per (layer, position, head) for keys and one for values.

    Symmetric, like the activation quantization (kernels.quantize_rows):
    when a token is stored, each head's 64 values get
    scale = max|x| / (2^(BITS-1) - 1) and codes round(x / scale). Reading
    applies the key's scale once per dot product (score) and folds the
    value's scale into the softmax weight (add_value), so there is no
    per-element scaling. A scale per head keeps one head with large values
    from costing the others their precision.
    """

    comptime DT = DType.int16 if Self.BITS == 16 else DType.int8
    comptime QMAX = Float32((1 << (Self.BITS - 1)) - 1)  # 32767 or 127

    var k: Pointer[Scalar[Self.DT], MutUntrackedOrigin]
    var v: Pointer[Scalar[Self.DT], MutUntrackedOrigin]
    var ks: FPtr  # key scales, [layer, position, head]
    var vs: FPtr  # value scales
    var n_layer: Int
    var max_t: Int
    var n_head: Int
    var head_dim: Int

    def __init__(
        out self,
        k: Pointer[Scalar[Self.DT], MutUntrackedOrigin],
        v: Pointer[Scalar[Self.DT], MutUntrackedOrigin],
        ks: FPtr,
        vs: FPtr,
        n_layer: Int,
        max_t: Int,
        n_head: Int,
        head_dim: Int,
    ):
        self.k = k
        self.v = v
        self.ks = ks
        self.vs = vs
        self.n_layer = n_layer
        self.max_t = max_t
        self.n_head = n_head
        self.head_dim = head_dim

    @staticmethod
    def name() -> String:
        return "int" + String(Self.BITS)

    @staticmethod
    def create(n_layer: Int, max_t: Int, n_head: Int, head_dim: Int) -> Self:
        comptime assert Self.BITS == 16 or Self.BITS == 8, "BITS must be 16 or 8"
        var n = n_layer * max_t * n_head * head_dim
        var ns = n_layer * max_t * n_head
        return Self(
            unsafe_alloc[Scalar[Self.DT]](n),
            unsafe_alloc[Scalar[Self.DT]](n),
            unsafe_alloc[Float32](ns),
            unsafe_alloc[Float32](ns),
            n_layer,
            max_t,
            n_head,
            head_dim,
        )

    @always_inline
    def slot(self, layer: Int, pos: Int, head: Int) -> Int:
        """Index of (layer, pos, head) among the scales; times head_dim, the
        offset of its values."""
        return (layer * self.max_t + pos) * self.n_head + head

    @staticmethod
    def quantize(x: FPtr, n: Int, dst: Pointer[Scalar[Self.DT], MutUntrackedOrigin]) -> Float32:
        """Quantizes x[n] into dst; returns the scale."""
        var mx = F32V(0)
        for i in range(0, n, NW):
            mx = max(mx, abs(x.unsafe_load[width=NW](i)))
        var s = mx.reduce_max() / Self.QMAX
        if s == 0:
            s = 1
        var inv = F32V(1 / s)
        for i in range(0, n, NW):
            dst.unsafe_store(i, round(x.unsafe_load[width=NW](i) * inv).cast[Self.DT]())
        return s

    def store(self, layer: Int, pos: Int, k: FPtr, v: FPtr):
        for h in range(self.n_head):
            var sl = self.slot(layer, pos, h)
            var o = h * self.head_dim
            self.ks[unsafe_offset=sl] = Self.quantize(
                k.unsafe_offset(o), self.head_dim, self.k.unsafe_offset(sl * self.head_dim)
            )
            self.vs[unsafe_offset=sl] = Self.quantize(
                v.unsafe_offset(o), self.head_dim, self.v.unsafe_offset(sl * self.head_dim)
            )

    @always_inline
    def score[HS: Int](self, layer: Int, pos: Int, head: Int, q: FPtr) -> Float32:
        var sl = self.slot(layer, pos, head)
        var kp = self.k.unsafe_offset(sl * HS)
        var d = F32V(0)
        comptime for i in range(0, HS, NW):
            d += q.unsafe_load[width=NW](i) * kp.unsafe_load[width=NW](i).cast[
                DType.float32
            ]()
        return d.reduce_add() * self.ks[unsafe_offset=sl]

    @always_inline
    def add_value[
        HS: Int
    ](
        self,
        layer: Int,
        pos: Int,
        head: Int,
        p: F32V,
        mut acc: Array[F32V, length = HS // NW],
    ):
        var sl = self.slot(layer, pos, head)
        var vp = self.v.unsafe_offset(sl * HS)
        var ps = p * F32V(self.vs[unsafe_offset=sl])
        comptime for i in range(HS // NW):
            acc[i] = ps.fma(vp.unsafe_load[width=NW](i * NW).cast[DType.float32](), acc[i])

    def nbytes(self) -> Int:
        var n = self.n_layer * self.max_t * self.n_head
        return 2 * n * (self.head_dim * Self.BITS // 8 + 4)

    def free(self):
        self.k.unsafe_free()
        self.v.unsafe_free()
        self.ks.unsafe_free()
        self.vs.unsafe_free()
