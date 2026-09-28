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
- QuantKV[BITS] (planned): int16 / int8 with a scale per position and head.
"""

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
