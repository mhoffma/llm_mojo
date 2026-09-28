"""CPU kernels for a GPT-2-style transformer, generic over the weight format.

Activations are always float32 (FPtr). Weight matrices are any `WeightMatrix`
and are read through `w.load[width](row, col)`, which widens to float32; the
arithmetic is float32 FMA. Small tensors (LayerNorm parameters, biases) stay
float32 pointers.

The kernels are the ones from gpt2.mojo with the weight pointer replaced by the
trait. With `DenseMatrix[float32]` they compile to the same loads and the same
order of floating-point operations, so they give bit-identical results.
"""

from std.math import exp, tanh, sqrt
from std.memory.alloc import unsafe_alloc
from std.runtime import parallelism_level
from max.algorithm import parallelize

from tensor import FPtr, NW, F32V, WeightMatrix

comptime MAX_PARTS = 64
"""Most threads gemv splits a matrix across; sizes its scratch buffer."""


@always_inline
def store_out[RESID: Bool, GELU: Bool](o: FPtr, off: Int, var r: F32V):
    """Applies the optional GELU, adds the residual, and stores."""
    comptime if GELU:
        comptime s = Float32(0.7978845608028654)  # sqrt(2/pi)
        r = 0.5 * r * (1 + tanh(s * (r + 0.044715 * r * r * r)))
    comptime if RESID:
        r += o.unsafe_load[width=NW](off)
    o.unsafe_store(off, r)


@always_inline
def mm_tile[
    W: WeightMatrix, TM: Int, NV: Int, RESID: Bool, GELU: Bool
](
    out_: FPtr,
    x: FPtr,
    w: W,
    b: FPtr,
    t0: Int,
    j0: Int,
    IN: Int,
    OUT: Int,
):
    """Computes a TM x (NV*NW) tile of out = x @ w + b, held in registers."""
    var acc = Array[F32V, length = TM * NV](fill=F32V(0))
    var wv = Array[F32V, length=NV](fill=F32V(0))
    for i in range(IN):
        comptime for v in range(NV):
            wv[v] = w.load[NW](i, j0 + v * NW)
        comptime for m in range(TM):
            var xm = F32V(x[unsafe_offset = (t0 + m) * IN + i])
            comptime for v in range(NV):
                acc[m * NV + v] = xm.fma(wv[v], acc[m * NV + v])
    comptime for m in range(TM):
        var orow = out_.unsafe_offset((t0 + m) * OUT + j0)
        comptime for v in range(NV):
            var r = acc[m * NV + v] + b.unsafe_load[width=NW](j0 + v * NW)
            store_out[RESID, GELU](orow, v * NW, r)


def matmul[
    W: WeightMatrix, //, RESID: Bool = False, GELU: Bool = False
](
    out_: FPtr,
    x: FPtr,
    w: W,
    b: FPtr,
    T: Int,
    IN: Int,
    OUT: Int,
    scratch: FPtr,
):
    """Computes out[T, OUT] (+)= act(x[T, IN] @ w[IN, OUT] + b[OUT]).

    Weights use the GPT-2 Conv1D layout [IN, OUT], so each input row
    contributes a contiguous SIMD-friendly slice of the output. For several
    rows (prefill), work is split across threads by column blocks, and rows are
    processed TM at a time so each weight load is reused TM times. For a single
    row (decode), see gemv.

    `W: WeightMatrix, //` makes W an *inferred* parameter: callers write
    `matmul[RESID=True](..., w, ...)` and W is taken from w's type.
    """
    if T == 1:
        gemv[RESID, GELU](out_, x, w, b, IN, OUT, scratch)
        return
    comptime NV = 2
    comptime TN = NV * NW
    comptime TM = 8
    debug_assert(OUT % TN == 0, "OUT must be a multiple of the tile width")

    def block(blk: Int) {imm}:
        var j0 = blk * TN
        var t = 0
        while t + TM <= T:
            mm_tile[W, TM, NV, RESID, GELU](out_, x, w, b, t, j0, IN, OUT)
            t += TM
        while t < T:
            mm_tile[W, 1, NV, RESID, GELU](out_, x, w, b, t, j0, IN, OUT)
            t += 1

    parallelize(block, OUT // TN)


def gemv[
    W: WeightMatrix, //, RESID: Bool, GELU: Bool
](out_: FPtr, x: FPtr, w: W, b: FPtr, IN: Int, OUT: Int, scratch: FPtr):
    """Single-row matmul, split over IN so each thread streams whole rows.

    Decoding is bound by memory bandwidth: every weight is read once per token.
    Giving each thread a contiguous band of rows keeps reads sequential; the
    per-thread partial sums (in scratch) are then reduced.
    """
    var nparts = min(parallelism_level(), MAX_PARTS)
    var rows = (IN + nparts - 1) // nparts

    def part(pi: Int) {imm}:
        var acc = scratch.unsafe_offset(pi * OUT)
        for j in range(0, OUT, NW):
            acc.unsafe_store(j, F32V(0))
        var i = pi * rows
        var end = min(IN, i + rows)
        while i + 4 <= end:
            var x0 = F32V(x[unsafe_offset=i])
            var x1 = F32V(x[unsafe_offset = i + 1])
            var x2 = F32V(x[unsafe_offset = i + 2])
            var x3 = F32V(x[unsafe_offset = i + 3])
            for j in range(0, OUT, NW):
                var s = acc.unsafe_load[width=NW](j)
                s = x0.fma(w.load[NW](i, j), s)
                s = x1.fma(w.load[NW](i + 1, j), s)
                s = x2.fma(w.load[NW](i + 2, j), s)
                s = x3.fma(w.load[NW](i + 3, j), s)
                acc.unsafe_store(j, s)
            i += 4
        while i < end:
            var xi = F32V(x[unsafe_offset=i])
            for j in range(0, OUT, NW):
                acc.unsafe_store(
                    j, xi.fma(w.load[NW](i, j), acc.unsafe_load[width=NW](j))
                )
            i += 1

    parallelize(part, nparts)
    for j in range(0, OUT, NW):
        var r = b.unsafe_load[width=NW](j)
        for pi in range(nparts):
            r += scratch.unsafe_load[width=NW](pi * OUT + j)
        store_out[RESID, GELU](out_, j, r)


def layernorm[C: Int](out_: FPtr, x: FPtr, w: FPtr, b: FPtr, T: Int):
    """LayerNorm over rows of C values (eps 1e-5, as in GPT-2)."""
    for t in range(T):
        var xr = x.unsafe_offset(t * C)
        var o = out_.unsafe_offset(t * C)
        var s = F32V(0)
        for i in range(0, C, NW):
            s += xr.unsafe_load[width=NW](i)
        var mean = s.reduce_add() / Float32(C)
        var sq = F32V(0)
        for i in range(0, C, NW):
            var d = xr.unsafe_load[width=NW](i) - mean
            sq += d * d
        var rstd = 1 / sqrt(sq.reduce_add() / Float32(C) + 1e-5)
        for i in range(0, C, NW):
            var n = (xr.unsafe_load[width=NW](i) - mean) * rstd
            o.unsafe_store(
                i, n * w.unsafe_load[width=NW](i) + b.unsafe_load[width=NW](i)
            )


def attention[
    N_HEAD: Int, HS: Int
](out_: FPtr, qkv: FPtr, kc: FPtr, vc: FPtr, T: Int, pos0: Int):
    """Causal multi-head attention for T new tokens at positions pos0.. .

    Queries come from qkv ([T, 3C]); keys and values come from the layer's KV
    cache ([pos, C]), which already holds positions 0 .. pos0+T-1.
    """
    comptime C = N_HEAD * HS
    var scale = 1 / sqrt(Float32(HS))

    def head_query(idx: Int) {imm}:
        var h = idx % N_HEAD
        var t = idx // N_HEAD
        var npos = pos0 + t + 1
        var q = qkv.unsafe_offset(t * 3 * C + h * HS)
        var scores = unsafe_alloc[Float32](npos)
        var mx = Float32.MIN
        for s in range(npos):
            var k = kc.unsafe_offset(s * C + h * HS)
            var d = F32V(0)
            comptime for i in range(0, HS, NW):
                d += q.unsafe_load[width=NW](i) * k.unsafe_load[width=NW](i)
            var sc = d.reduce_add() * scale
            scores[unsafe_offset=s] = sc
            mx = max(mx, sc)
        var total = Float32(0)
        for s in range(npos):
            var e = exp(scores[unsafe_offset=s] - mx)
            scores[unsafe_offset=s] = e
            total += e
        var o = out_.unsafe_offset(t * C + h * HS)
        var acc = Array[F32V, length = HS // NW](fill=F32V(0))
        for s in range(npos):
            var p = F32V(scores[unsafe_offset=s] / total)
            var v = vc.unsafe_offset(s * C + h * HS)
            comptime for i in range(HS // NW):
                acc[i] = p.fma(v.unsafe_load[width=NW](i * NW), acc[i])
        comptime for i in range(HS // NW):
            o.unsafe_store(i * NW, acc[i])
        scores.unsafe_free()

    # Waking the worker threads costs more than a short-context decode step's
    # attention, so only go parallel when there is enough work.
    if T * (pos0 + T) < 256:
        for i in range(N_HEAD * T):
            head_query(i)
    else:
        parallelize(head_query, N_HEAD * T)


def lm_head[W: WeightMatrix, //](logits: FPtr, h: FPtr, wte: W, V: Int, C: Int):
    """Computes logits[V] = h[C] @ wte[V, C]^T (the output head is tied to wte).
    """
    comptime CHUNK = 512
    var nchunks = (V + CHUNK - 1) // CHUNK

    def chunk(ci: Int) {imm}:
        var end = min(V, (ci + 1) * CHUNK)
        for v in range(ci * CHUNK, end):
            var d = F32V(0)
            for i in range(0, C, NW):
                d = h.unsafe_load[width=NW](i).fma(wte.load[NW](v, i), d)
            logits[unsafe_offset=v] = d.reduce_add()

    parallelize(chunk, nchunks)


def lm_head_rows[
    W: WeightMatrix, //
](logits: FPtr, h: FPtr, wte: W, T: Int, V: Int, C: Int):
    """Computes logits[T, V] = h[T, C] @ wte[V, C]^T, for every position.

    Used for evaluation (perplexity), which needs a prediction at each
    position, not just the last. Each wte row is loaded once and dotted with
    4 rows of h at a time.
    """
    comptime CHUNK = 64
    var nchunks = (V + CHUNK - 1) // CHUNK

    def chunk(ci: Int) {imm}:
        var end = min(V, (ci + 1) * CHUNK)
        for v in range(ci * CHUNK, end):
            var t = 0
            while t + 4 <= T:
                var d0 = F32V(0)
                var d1 = F32V(0)
                var d2 = F32V(0)
                var d3 = F32V(0)
                var h0 = h.unsafe_offset(t * C)
                for i in range(0, C, NW):
                    var w = wte.load[NW](v, i)
                    d0 = h0.unsafe_load[width=NW](i).fma(w, d0)
                    d1 = h0.unsafe_load[width=NW](C + i).fma(w, d1)
                    d2 = h0.unsafe_load[width=NW](2 * C + i).fma(w, d2)
                    d3 = h0.unsafe_load[width=NW](3 * C + i).fma(w, d3)
                logits[unsafe_offset = t * V + v] = d0.reduce_add()
                logits[unsafe_offset = (t + 1) * V + v] = d1.reduce_add()
                logits[unsafe_offset = (t + 2) * V + v] = d2.reduce_add()
                logits[unsafe_offset = (t + 3) * V + v] = d3.reduce_add()
                t += 4
            while t < T:
                var d = F32V(0)
                var ht = h.unsafe_offset(t * C)
                for i in range(0, C, NW):
                    d = ht.unsafe_load[width=NW](i).fma(wte.load[NW](v, i), d)
                logits[unsafe_offset = t * V + v] = d.reduce_add()
                t += 1

    parallelize(chunk, nchunks)
