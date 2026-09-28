"""Integer attention: an int8 KV cache laid out for VNNI, with attention
computed in integers (PLAN.md, M5).

Selected with `--kv int8 --attention int`. The model calls attention the
same way as for any cache (kernels.attention -> KVCache.attend); only the
cache type differs.

For one query token and head (npos = positions 0 .. pos):

 1. Query -> int16, symmetric: sq = max|q| / 32767, qi = round(q / sq).
 2. Scores s_int = qi · k for 16 positions at once. Keys are int8, stored
    per head in blocks of 16 positions, in pairs along the head dimension:
    [pos / 16][d / 2][16 positions][2]. One broadcast pair (qi[d], qi[d+1])
    and one VPDPWSSD add 2 dimensions of 16 positions' scores. With int8
    keys a 64-dimension score is at most 64 * 32767 * 127 = 2.7e8, so int32
    can't overflow (full-range int16 keys could).
 3. One shared scale. Each key has its own scale sk (as an integer kq =
    sk * 2^24), so raw scores aren't comparable across positions. They are
    rescaled with integer multiplies into fixed point with 16 fraction bits:
    t = (s_int * kq) >> 24  ~ s_int * sk
    S = (t * qm) >> 24      with qm = sq / sqrt(64) * 2^40
    so S = score * 2^16 for every position.
 4. intmath.masked_exp on S (positions >= npos masked): the unnormalized
    softmax e_i in Q30, the largest exactly 1.0, and their sum E. Not
    normalizing yet keeps the weights precise: with Q15 probabilities, a
    flat attention over ~1000 positions had only ~30 steps per weight, and
    errors up to ~2% of the output (tests/int_attention_check.mojo).
 5. Values: each value has its own scale sv (as vq = sv * 2^24). The scales
    are folded into the weights, relative to the largest in the head:
    w = e * (vq / vq_max) >> 15, int16 (<= 32767). Values are int8 stored per
    head in position pairs, [pos / 2][d][2], so one broadcast pair
    (w[s], w[s+1]) and one VPDPWSSD add 2 positions to 16 dimensions. A sum
    can reach 256 * 32767 * 127 < 2^31 over 256 positions, so the int32 sums
    move to float every 256 positions.
 6. out = sums * (vq_max / 2^24) * 2^15 / E: the softmax's normalization and
    the value scale, as one float multiply per output value (attention's
    output feeds the next matmul, which takes float32).

Positions past npos in a 16-position key block or a value pair are never
initialized; they get probability 0 (masked in the softmax, weight 0), so
they don't affect the result.
"""

from std.math import round, sqrt
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.os import abort
from max.algorithm import parallelize

from tensor import FPtr, NW, F32V, F32x16, I16x32, I32x16, dot_pairs
from intmath import masked_exp, exp_multiplier
from kvcache import KVCache

comptime KS = 24  # fraction bits of the integer key and value scales
comptime SF = 16  # fraction bits of the fixed-point scores
comptime PB = 16  # positions per key block (one int32 lane each)

comptime I8Ptr = Pointer[Int8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]


struct IntAttnKV(KVCache):
    """int8 keys and values in VNNI layouts, with integer attention."""

    var k: I8Ptr  # per (layer, head): [max_t / 16][head_dim / 2][16][2]
    var v: I8Ptr  # per (layer, head): [max_t / 2][head_dim][2]
    var kq: I32Ptr  # key scales * 2^24, per (layer, head, position)
    var vq: I32Ptr  # value scales * 2^24
    var n_layer: Int
    var max_t: Int
    var n_head: Int
    var head_dim: Int

    def __init__(
        out self,
        k: I8Ptr,
        v: I8Ptr,
        kq: I32Ptr,
        vq: I32Ptr,
        n_layer: Int,
        max_t: Int,
        n_head: Int,
        head_dim: Int,
    ):
        self.k = k
        self.v = v
        self.kq = kq
        self.vq = vq
        self.n_layer = n_layer
        self.max_t = max_t
        self.n_head = n_head
        self.head_dim = head_dim

    @staticmethod
    def name() -> String:
        return "int8-intattn"

    @staticmethod
    def create(n_layer: Int, max_t: Int, n_head: Int, head_dim: Int) -> Self:
        if max_t % PB != 0 or head_dim % 32 != 0:
            abort_bad_shape()
        var n = n_layer * n_head * max_t * head_dim
        var ns = n_layer * n_head * max_t
        return Self(
            unsafe_alloc[Int8](n),
            unsafe_alloc[Int8](n),
            unsafe_alloc[Int32](ns),
            unsafe_alloc[Int32](ns),
            n_layer,
            max_t,
            n_head,
            head_dim,
        )

    @always_inline
    def lh(self, layer: Int, head: Int) -> Int:
        """Index of (layer, head)."""
        return layer * self.n_head + head

    @always_inline
    def k_at(self, layer: Int, head: Int, pos: Int, d: Int) -> Int:
        """Offset of key element (pos, d) in the blocked layout."""
        var blk = self.lh(layer, head) * (self.max_t // PB) + pos // PB
        return ((blk * (self.head_dim // 2) + d // 2) * PB + pos % PB) * 2 + d % 2

    @always_inline
    def v_at(self, layer: Int, head: Int, pos: Int, d: Int) -> Int:
        """Offset of value element (pos, d) in the position-pair layout."""
        var pair = self.lh(layer, head) * (self.max_t // 2) + pos // 2
        return (pair * self.head_dim + d) * 2 + pos % 2

    def store(self, layer: Int, pos: Int, k: FPtr, v: FPtr):
        for h in range(self.n_head):
            var kh = k.unsafe_offset(h * self.head_dim)
            var vh = v.unsafe_offset(h * self.head_dim)
            var sl = self.lh(layer, h) * self.max_t + pos
            self.kq[unsafe_offset=sl] = self.quantize_head(kh, layer, h, pos, True)
            self.vq[unsafe_offset=sl] = self.quantize_head(vh, layer, h, pos, False)

    def quantize_head(self, x: FPtr, layer: Int, h: Int, pos: Int, key: Bool) -> Int32:
        """Quantizes one head's values to int8 into the key or value layout;
        returns the scale * 2^24.

        16 values at a time: keys are written as int16 pairs (dimensions d,
        d+1 are adjacent in the key layout, and successive pairs are PB * 2
        bytes apart); values are written at a stride of 2 bytes (the two
        positions of a pair are interleaved)."""
        var mx = F32V(0)
        for d in range(0, self.head_dim, NW):
            mx = max(mx, abs(x.unsafe_load[width=NW](d)))
        var s = mx.reduce_max() / 127
        if s == 0:
            s = 1
        var inv = F32V(1 / s)
        if key:
            var kp = self.k.unsafe_offset(self.k_at(layer, h, pos, 0)).unsafe_bitcast[Int16]()
            for d in range(0, self.head_dim, NW):
                var c = round(x.unsafe_load[width=NW](d) * inv).cast[DType.int8]()
                var pairs = bitcast[DType.int16, NW // 2](c)
                comptime for j in range(NW // 2):
                    kp[unsafe_offset = (d // 2 + j) * PB] = pairs[j]
        else:
            var vp = self.v.unsafe_offset(self.v_at(layer, h, pos, 0))
            for d in range(0, self.head_dim, NW):
                var c = round(x.unsafe_load[width=NW](d) * inv).cast[DType.int8]()
                comptime for j in range(NW):
                    vp[unsafe_offset = 2 * (d + j)] = c[j]
        return Int32(Int(round(Float64(s) * Float64(1 << KS))))

    def attend_one[
        N_HEAD: Int, HS: Int
    ](self, out_: FPtr, qkv: FPtr, layer: Int, t: Int, h: Int, pos0: Int):
        comptime C = N_HEAD * HS
        var mult = exp_multiplier(1.0 / Float64(1 << SF))
        var npos = pos0 + t + 1
        var nr = (npos + PB - 1) // PB * PB
        var q = qkv.unsafe_offset(t * 3 * C + h * HS)
        var qi = unsafe_alloc[Int16](HS)
        var scores = unsafe_alloc[Int32](nr)
        var e = unsafe_alloc[Int32](nr)
        var w = unsafe_alloc[Int16](nr + 2)

        # 1. Query -> int16.
        var mq = F32V(0)
        for d in range(0, HS, NW):
            mq = max(mq, abs(q.unsafe_load[width=NW](d)))
        var sq = mq.reduce_max() / 32767
        if sq == 0:
            sq = 1
        var invq = F32V(1 / sq)
        for d in range(0, HS, NW):
            qi.unsafe_store(
                d, round(q.unsafe_load[width=NW](d) * invq).cast[DType.int16]()
            )
        var qm = Int(
            round(Float64(sq) / sqrt(Float64(HS)) * Float64(1 << (SF + KS)))
        )
        var qm64 = SIMD[DType.int64, PB](qm)

        # 2-3. Scores for 16 positions per block, then rescaled to 16
        # fraction bits.
        var q32 = qi.unsafe_bitcast[Int32]()  # query pairs
        var ks = self.kq.unsafe_offset(self.lh(layer, h) * self.max_t)
        for b in range(nr // PB):
            var acc = I32x16(0)
            var kb = self.k.unsafe_offset(self.k_at(layer, h, b * PB, 0))
            comptime for j in range(HS // 2):
                var qp = bitcast[DType.int16, 32](I32x16(q32[unsafe_offset=j]))
                var kv = kb.unsafe_load[width=32](j * 2 * PB).cast[DType.int16]()
                acc = dot_pairs(acc, qp, kv)
            var kqv = ks.unsafe_load[width=PB](b * PB).cast[DType.int64]()
            var t1 = (acc.cast[DType.int64]() * kqv + (1 << (KS - 1))) >> KS
            var sfix = (t1 * qm64 + (1 << (KS - 1))) >> KS
            scores.unsafe_store(b * PB, sfix.cast[DType.int32]())

        # 4. Unnormalized integer softmax over positions 0 .. npos-1.
        var total = masked_exp[False](scores, scores, npos, mult, e)

        # 5. Fold the value scales into the weights, relative to the
        # largest one in this head.
        var vs = self.vq.unsafe_offset(self.lh(layer, h) * self.max_t)
        var vmax = 0
        for s in range(npos):
            vmax = max(vmax, Int(vs[unsafe_offset=s]))
        var o = out_.unsafe_offset(t * C + h * HS)
        if vmax == 0 or total == 0:
            for d in range(0, HS, NW):
                o.unsafe_store(d, F32V(0))
        else:
            var inv47 = SIMD[DType.int64, PB]((1 << 47) // vmax)
            for b in range(nr // PB):
                var vv = vs.unsafe_load[width=PB](b * PB).cast[DType.int64]()
                var r16 = (vv * inv47) >> 31  # vq / vq_max in Q16
                var ee = e.unsafe_load[width=PB](b * PB).cast[DType.int64]()
                # e (Q30) * r (Q16) >> 31: Q15, at most 32768.
                var ww = min((ee * r16 + (1 << 30)) >> 31, SIMD[DType.int64, PB](32767))
                # Positions past npos have e = 0; their scale may be
                # anything (uninitialized), so zero them explicitly.
                w.unsafe_store(b * PB, ee.eq(0).select(SIMD[DType.int64, PB](0), ww).cast[DType.int16]())
            w[unsafe_offset=nr] = 0
            w[unsafe_offset = nr + 1] = 0

            # Weighted sum of values, 2 positions per VPDPWSSD; the int32
            # sums move to float every 256 positions (128 pairs).
            var accf = Array[F32x16, length = HS // 16](fill=F32x16(0))
            var w32 = w.unsafe_bitcast[Int32]()
            var npairs = (npos + 1) // 2
            for seg in range(0, npairs, 128):
                var accv = Array[I32x16, length = HS // 16](fill=I32x16(0))
                for pp2 in range(seg, min(npairs, seg + 128)):
                    var wp = bitcast[DType.int16, 32](I32x16(w32[unsafe_offset=pp2]))
                    var vb = self.v.unsafe_offset(self.v_at(layer, h, 2 * pp2, 0))
                    comptime for c in range(HS // 16):
                        var vv8 = vb.unsafe_load[width=32](c * 32).cast[DType.int16]()
                        accv[c] = dot_pairs(accv[c], wp, vv8)
                comptime for c in range(HS // 16):
                    accf[c] += accv[c].cast[DType.float32]()

            # 6. Back to float: * (vq_max / 2^24) * 2^15 / E.
            var fs = F32x16(
                Float32(Float64(vmax) * Float64(1 << 15) / (Float64(1 << KS) * Float64(total)))
            )
            comptime for c in range(HS // 16):
                o.unsafe_store(c * 16, accf[c] * fs)

        qi.unsafe_free()
        scores.unsafe_free()
        e.unsafe_free()
        w.unsafe_free()


    def nbytes(self) -> Int:
        var n = self.n_layer * self.n_head * self.max_t
        return 2 * n * (self.head_dim + 4)

    def free(self):
        self.k.unsafe_free()
        self.v.unsafe_free()
        self.kq.unsafe_free()
        self.vq.unsafe_free()


def abort_bad_shape():
    abort("IntAttnKV needs max_t % 16 == 0 and head_dim % 32 == 0")
