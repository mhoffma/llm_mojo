"""CPU kernels for a GPT-2-style transformer, generic over the weight format.

Activations are always float32 (FPtr). Weight matrices are any `WeightMatrix`
and are read through `w.load[width](row, col)`, which widens to float32; the
arithmetic is float32 FMA. Small tensors (LayerNorm parameters, biases) stay
float32 pointers.

The kernels are the ones from gpt2.mojo with the weight pointer replaced by the
trait. With `DenseMatrix[float32]` they compile to the same loads and the same
order of floating-point operations, so they give bit-identical results.
"""

from std.math import exp, tanh, sqrt, round
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.runtime import parallelism_level
from max.algorithm import parallelize

from kvcache import KVCache
from team import Team, split
from tensor import (
    FPtr,
    NW,
    F32V,
    F32x16,
    I16Ptr,
    I16x32,
    I32x16,
    WeightMatrix,
    DenseMatrix,
    dot_pairs,
)

comptime MAX_PARTS = 64
"""Most threads gemv splits a matrix across; sizes its scratch buffer."""


@always_inline
def store_out[
    RESID: Bool, GELU: Bool, width: Int = NW
](o: FPtr, off: Int, var r: SIMD[DType.float32, width]):
    """Applies the optional GELU, adds the residual, and stores."""
    comptime if GELU:
        comptime s = Float32(0.7978845608028654)  # sqrt(2/pi)
        r = 0.5 * r * (1 + tanh(s * (r + 0.044715 * r * r * r)))
    comptime if RESID:
        r += o.unsafe_load[width=width](off)
    o.unsafe_store(off, r)


@always_inline
def mm_tile[
    W: WeightMatrix,
    TM: Int,
    NV: Int,
    RESID: Bool,
    GELU: Bool,
    BIAS: Bool = True,
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
            var r = acc[m * NV + v]
            comptime if BIAS:
                r += b.unsafe_load[width=NW](j0 + v * NW)
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
    KV: KVCache, //, N_HEAD: Int, HS: Int
](out_: FPtr, qkv: FPtr, kv: KV, layer: Int, T: Int, pos0: Int):
    """Causal multi-head attention for T new tokens at positions pos0.. .

    Queries come from qkv ([T, 3C]); keys and values come from the KV cache,
    which already holds this layer's positions 0 .. pos0+T-1. How attention
    is computed belongs to the cache format (KVCache.attend): float formats
    share kvcache.attend_float, and IntAttnKV computes in integers.
    """
    kv.attend[N_HEAD, HS](out_, qkv, layer, T, pos0)


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


@always_inline
def finish[RESID: Bool, GELU: Bool](o: FPtr, i: Int, var r: Float32):
    """Scalar version of store_out: optional GELU, residual add, store."""
    comptime if GELU:
        comptime s = Float32(0.7978845608028654)  # sqrt(2/pi)
        r = 0.5 * r * (1 + tanh(s * (r + 0.044715 * r * r * r)))
    comptime if RESID:
        r += o[unsafe_offset=i]
    o[unsafe_offset=i] = r


def matmul_rows[
    W: WeightMatrix, //, RESID: Bool = False, GELU: Bool = False, BIAS: Bool = True
](out_: FPtr, x: FPtr, w: W, b: FPtr, T: Int, IN: Int, OUT: Int):
    """Computes out[T, OUT] (+)= act(x[T, IN] @ w[OUT, IN]^T + b[OUT]).

    For weights stored one output per row (W.OUT_MAJOR), as GGUF files store
    them. Two strategies:

    - One token (decode): threads split the output rows, and each row's dot
      product with x is computed while dequantizing (w.dot_row), without
      writing the float32 weights anywhere.
    - Several tokens (prefill): threads take 32 output rows at a time,
      dequantize them into a float32 tile transposed to [IN, 32], and run the
      float32 tile kernel (mm_tile) over all T tokens. Dequantizing is done
      once per weight and shared by every token.
    """
    if T == 1:
        comptime RB = 16  # output rows per task

        def one(ti: Int) {imm}:
            for o in range(ti * RB, min(OUT, (ti + 1) * RB)):
                var r = w.dot_row(o, x)
                comptime if BIAS:
                    r += b[unsafe_offset=o]
                finish[RESID, GELU](out_, o, r)

        parallelize(one, (OUT + RB - 1) // RB)
        return

    comptime NV = 2
    comptime TN = NV * NW  # output rows per tile
    comptime TM = 8
    var full = OUT // TN
    var ntasks = full + (1 if OUT % TN != 0 else 0)

    def tile_task(ti: Int) {imm}:
        var row = unsafe_alloc[Float32](IN)
        if ti == full:  # the last OUT % TN rows (the vocabulary, 50257)
            rows_dot[RESID, GELU, BIAS](out_, x, w, b, T, IN, OUT, full * TN, OUT, row)
            row.unsafe_free()
            return
        var o0 = ti * TN
        var tile = unsafe_alloc[Float32](IN * TN)
        for r in range(TN):
            w.dequant_row(o0 + r, row)
            for i in range(IN):
                tile[unsafe_offset = i * TN + r] = row[unsafe_offset=i]
        # mm_tile reads w.load(i, j) for output columns j = o0 .. o0+TN-1;
        # shifting the view by -o0 maps them to the tile's columns 0 .. TN-1.
        var view = DenseMatrix[DType.float32](tile.unsafe_offset(-o0), IN, TN)
        var t = 0
        while t + TM <= T:
            mm_tile[DenseMatrix[DType.float32], TM, NV, RESID, GELU, BIAS](
                out_, x, view, b, t, o0, IN, OUT
            )
            t += TM
        while t < T:
            mm_tile[DenseMatrix[DType.float32], 1, NV, RESID, GELU, BIAS](
                out_, x, view, b, t, o0, IN, OUT
            )
            t += 1
        tile.unsafe_free()
        row.unsafe_free()

    parallelize(tile_task, ntasks)


def rows_dot[
    W: WeightMatrix, //, RESID: Bool, GELU: Bool, BIAS: Bool
](
    out_: FPtr,
    x: FPtr,
    w: W,
    b: FPtr,
    T: Int,
    IN: Int,
    OUT: Int,
    o_start: Int,
    o_end: Int,
    row: FPtr,
):
    """Output rows o_start..o_end-1, one at a time: dequantize the row into
    `row`, then dot it with 4 rows of x at a time."""
    for o in range(o_start, o_end):
        w.dequant_row(o, row)
        var bias = Float32(0)
        comptime if BIAS:
            bias = b[unsafe_offset=o]
        var t = 0
        while t + 4 <= T:
            var d0 = F32V(0)
            var d1 = F32V(0)
            var d2 = F32V(0)
            var d3 = F32V(0)
            var x0 = x.unsafe_offset(t * IN)
            for i in range(0, IN, NW):
                var wv = row.unsafe_load[width=NW](i)
                d0 = x0.unsafe_load[width=NW](i).fma(wv, d0)
                d1 = x0.unsafe_load[width=NW](IN + i).fma(wv, d1)
                d2 = x0.unsafe_load[width=NW](2 * IN + i).fma(wv, d2)
                d3 = x0.unsafe_load[width=NW](3 * IN + i).fma(wv, d3)
            finish[RESID, GELU](out_, t * OUT + o, d0.reduce_add() + bias)
            finish[RESID, GELU](out_, (t + 1) * OUT + o, d1.reduce_add() + bias)
            finish[RESID, GELU](out_, (t + 2) * OUT + o, d2.reduce_add() + bias)
            finish[RESID, GELU](out_, (t + 3) * OUT + o, d3.reduce_add() + bias)
            t += 4
        while t < T:
            var d = F32V(0)
            var xt = x.unsafe_offset(t * IN)
            for i in range(0, IN, NW):
                d = xt.unsafe_load[width=NW](i).fma(row.unsafe_load[width=NW](i), d)
            finish[RESID, GELU](out_, t * OUT + o, d.reduce_add() + bias)
            t += 1


def quantize_rows(x: FPtr, T: Int, IN: Int, xq: I16Ptr, sx: FPtr):
    """Quantizes each row of x[T, IN] to int16, symmetric, with one scale per
    row: sx[t] = max|x[t]| / 32767 and xq[t] = round(x[t] / sx[t])."""
    for t in range(T):
        var row = x.unsafe_offset(t * IN)
        var mx = F32V(0)
        for i in range(0, IN, NW):
            mx = max(mx, abs(row.unsafe_load[width=NW](i)))
        var s = mx.reduce_max() / 32767
        if s == 0:
            s = 1
        sx[unsafe_offset=t] = s
        var inv = F32V(1 / s)
        var dst = xq.unsafe_offset(t * IN)
        for i in range(0, IN, NW):
            var q = round(row.unsafe_load[width=NW](i) * inv)
            dst.unsafe_store(i, q.cast[DType.int16]())


def matmul_rows_a16[
    W: WeightMatrix, //, RESID: Bool = False, GELU: Bool = False, BIAS: Bool = True
](out_: FPtr, x: FPtr, w: W, b: FPtr, T: Int, IN: Int, OUT: Int):
    """Computes out[T, OUT] (+)= act(x[T, IN] @ w[OUT, IN]^T + b[OUT]) in
    integers: W4A16 / W8A16, for ACT16 formats (QuantMatrix[..., A16=True]).

    x is first quantized to int16 per row (quantize_rows). Then, per output
    row and quantization group, int16 activations times int16 weight codes
    (minus the zero point) are summed in int32 with VPDPWSSD; each group's
    sum takes the group's scale, and the row's result the activations'
    scale.

    - One token (decode): threads split the output rows; w.dot_row_i16
      unpacks and multiplies in one pass.
    - Several tokens (prefill): threads take 32 output rows at a time, unpack
      them to int16 in the VNNI layout, and run register tiles of 4 tokens x
      32 outputs over all tokens (tile_i16), so group scales are applied
      once per group per tile.
    """
    var xq = unsafe_alloc[Int16](T * IN)
    var sx = unsafe_alloc[Float32](T)
    quantize_rows(x, T, IN, xq, sx)
    comptime RB = 16  # output rows per task

    if T == 1:
        # Formats with their own decode layout (QuantMatrix BLOCKED) get the
        # activations permuted to match, once per token.
        var xp = unsafe_alloc[Int16](IN)
        var xs = unsafe_alloc[Float32](IN // 16)
        var x1 = xp if w.permute_x_i16(xq, IN, xp, xs) else xq

        def one(ti: Int) {imm}:
            var s = sx[unsafe_offset=0]
            for o in range(ti * RB, min(OUT, (ti + 1) * RB)):
                var r = w.dot_row_i16(o, x1, xs) * s
                comptime if BIAS:
                    r += b[unsafe_offset=o]
                finish[RESID, GELU](out_, o, r)

        parallelize(one, (OUT + RB - 1) // RB)
        xp.unsafe_free()
        xs.unsafe_free()
    else:
        var G = w.group_size()
        comptime NV = 2
        comptime TN = NV * 16  # output rows per tile
        comptime TM = 4  # tokens per register tile
        var full = OUT // TN
        var ntasks = full + (1 if OUT % TN != 0 else 0)

        def tile_task(ti: Int) {imm}:
            if ti == full:  # the last OUT % TN rows (the vocabulary, 50257)
                rows_i16[RESID, GELU, BIAS](
                    out_, xq, sx, w, b, T, IN, OUT, G, full * TN, OUT
                )
                return
            var o0 = ti * TN
            # Unpack TN output rows into the VNNI layout: for each pair of
            # inputs (k, k+1), the TN outputs' weights as int16 pairs, so one
            # 32-lane load holds 16 outputs x 2 inputs.
            var wt = unsafe_alloc[Int16](IN * TN)
            var st = unsafe_alloc[Float32](IN // G * TN)  # [group][output]
            var row = unsafe_alloc[Int16](IN)
            var rs = unsafe_alloc[Float32](IN // G)
            for j in range(TN):
                w.unpack_row_i16(o0 + j, row, rs)
                for k in range(IN):
                    wt[unsafe_offset = ((k // 2) * TN + j) * 2 + k % 2] = row[
                        unsafe_offset=k
                    ]
                for g in range(IN // G):
                    st[unsafe_offset = g * TN + j] = rs[unsafe_offset=g]
            var t = 0
            while t + TM <= T:
                tile_i16[TM, NV, RESID, GELU, BIAS](
                    out_, xq, sx, wt, st, b, t, o0, IN, OUT, G
                )
                t += TM
            while t < T:
                tile_i16[1, NV, RESID, GELU, BIAS](
                    out_, xq, sx, wt, st, b, t, o0, IN, OUT, G
                )
                t += 1
            wt.unsafe_free()
            st.unsafe_free()
            row.unsafe_free()
            rs.unsafe_free()

        parallelize(tile_task, ntasks)
    xq.unsafe_free()
    sx.unsafe_free()


@always_inline
def tile_i16[
    TM: Int, NV: Int, RESID: Bool, GELU: Bool, BIAS: Bool
](
    out_: FPtr,
    xq: I16Ptr,
    sx: FPtr,
    wt: I16Ptr,
    st: FPtr,
    b: FPtr,
    t0: Int,
    o0: Int,
    IN: Int,
    OUT: Int,
    G: Int,
):
    """Computes a TM x (NV*16) tile of the integer matmul, in registers.

    wt holds NV*16 output rows in the VNNI layout (see matmul_rows_a16) and
    st their group scales as [group][output]. For each input pair (k, k+1),
    token m's two int16 activations are broadcast as one int32 to all 16
    lanes, and one VPDPWSSD per NV adds x(k)*w(o,k) + x(k+1)*w(o,k+1) for 16
    outputs o. At the end of each group, the int32 sums become float and
    take the group's scale, for the whole tile at once.

    Here one int32 lane sums a whole group for one output, so the sums are
    moved to float at least every SEG inputs: 256 products of at most
    32767 * 255 (int16 activation times int8 code minus zero point) is
    2.139e9, just under 2^31. The scale is the same across a group, so where
    a group is split doesn't change the result.
    """
    comptime TN = NV * 16
    comptime SEG = 256
    var x32 = xq.unsafe_bitcast[Int32]()  # activation pairs, one per int32
    var accf = Array[F32x16, length = TM * NV](fill=F32x16(0))
    var seg = min(G, SEG)
    for k0 in range(0, IN, seg):
        var g = k0 // G
        var acc = Array[I32x16, length = TM * NV](fill=I32x16(0))
        for kp in range(k0 // 2, (k0 + seg) // 2):
            var wv = Array[I16x32, length=NV](fill=I16x32(0))
            comptime for v in range(NV):
                wv[v] = wt.unsafe_load[width=32]((kp * TN + v * 16) * 2)
            comptime for m in range(TM):
                var pair = x32[unsafe_offset = (t0 + m) * (IN // 2) + kp]
                var xb = bitcast[DType.int16, 32](I32x16(pair))
                comptime for v in range(NV):
                    acc[m * NV + v] = dot_pairs(acc[m * NV + v], xb, wv[v])
        comptime for v in range(NV):
            var s = st.unsafe_load[width=16](g * TN + v * 16)
            comptime for m in range(TM):
                accf[m * NV + v] = acc[m * NV + v].cast[DType.float32]().fma(
                    s, accf[m * NV + v]
                )
    comptime for m in range(TM):
        var sxm = F32x16(sx[unsafe_offset = t0 + m])
        var orow = out_.unsafe_offset((t0 + m) * OUT + o0)
        comptime for v in range(NV):
            var r = accf[m * NV + v] * sxm
            comptime if BIAS:
                r += b.unsafe_load[width=16](o0 + v * 16)
            store_out[RESID, GELU, 16](orow, v * 16, r)


def rows_i16[
    W: WeightMatrix, //, RESID: Bool, GELU: Bool, BIAS: Bool
](
    out_: FPtr,
    xq: I16Ptr,
    sx: FPtr,
    w: W,
    b: FPtr,
    T: Int,
    IN: Int,
    OUT: Int,
    G: Int,
    o_start: Int,
    o_end: Int,
):
    """Output rows o_start..o_end-1 one at a time: unpack the row to int16,
    then dot it with 4 tokens at a time. For the rows left over after the
    tiles."""
    var wq = unsafe_alloc[Int16](IN)
    var ws = unsafe_alloc[Float32](IN // G)
    for o in range(o_start, o_end):
        w.unpack_row_i16(o, wq, ws)
        var bias = Float32(0)
        comptime if BIAS:
            bias = b[unsafe_offset=o]
        var t = 0
        while t + 4 <= T:
            var d = dot4_i16(wq, ws, xq.unsafe_offset(t * IN), IN, G)
            comptime for k in range(4):
                finish[RESID, GELU](
                    out_, (t + k) * OUT + o, d[k] * sx[unsafe_offset = t + k] + bias
                )
            t += 4
        while t < T:
            var d = dot1_i16(wq, ws, xq.unsafe_offset(t * IN), IN, G)
            finish[RESID, GELU](out_, t * OUT + o, d * sx[unsafe_offset=t] + bias)
            t += 1
    wq.unsafe_free()
    ws.unsafe_free()


@always_inline
def dot1_i16(wq: I16Ptr, ws: FPtr, xq: I16Ptr, IN: Int, G: Int) -> Float32:
    """sum over groups of ws[g] * sum(wq * xq) for one row of activations."""
    var accf = F32x16(0)
    for g in range(IN // G):
        var acc = I32x16(0)
        for k in range(g * G, (g + 1) * G, 32):
            acc = dot_pairs(
                acc, xq.unsafe_load[width=32](k), wq.unsafe_load[width=32](k)
            )
        accf = acc.cast[DType.float32]().fma(F32x16(ws[unsafe_offset=g]), accf)
    return accf.reduce_add()


@always_inline
def dot4_i16(
    wq: I16Ptr, ws: FPtr, xq: I16Ptr, IN: Int, G: Int
) -> SIMD[DType.float32, 4]:
    """dot1_i16 for 4 consecutive activation rows at once, loading each
    weight vector once for all 4."""
    var accf = Array[F32x16, length=4](fill=F32x16(0))
    for g in range(IN // G):
        var acc = Array[I32x16, length=4](fill=I32x16(0))
        for k in range(g * G, (g + 1) * G, 32):
            var wv = wq.unsafe_load[width=32](k)
            comptime for r in range(4):
                acc[r] = dot_pairs(
                    acc[r], xq.unsafe_load[width=32](r * IN + k), wv
                )
        var s = F32x16(ws[unsafe_offset=g])
        comptime for r in range(4):
            accf[r] = acc[r].cast[DType.float32]().fma(s, accf[r])
    var out = SIMD[DType.float32, 4](0)
    comptime for r in range(4):
        out[r] = accf[r].reduce_add()
    return out


def linear[
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
    """out[T, OUT] (+)= act(x @ w + b), with the kernel for w's layout."""
    comptime if W.ACT16:
        matmul_rows_a16[RESID=RESID, GELU=GELU](out_, x, w, b, T, IN, OUT)
    elif W.OUT_MAJOR:
        matmul_rows[RESID=RESID, GELU=GELU](out_, x, w, b, T, IN, OUT)
    else:
        matmul[RESID=RESID, GELU=GELU](out_, x, w, b, T, IN, OUT, scratch)


def head[W: WeightMatrix, //](logits: FPtr, h: FPtr, w: W, T: Int, V: Int, C: Int):
    """logits[T, V] = h[T, C] @ w[V, C]^T: the output head, for T rows."""
    comptime if W.ACT16:
        matmul_rows_a16[BIAS=False](logits, h, w, h, T, C, V)
    elif W.OUT_MAJOR:
        matmul_rows[BIAS=False](logits, h, w, h, T, C, V)
    else:
        if T == 1:
            lm_head(logits, h, w, V, C)
        else:
            lm_head_rows(logits, h, w, T, V, C)


# ===----------------------------------------------------------------------=== #
# One-token kernels for a persistent team of threads (team.mojo)
# ===----------------------------------------------------------------------=== #
#
# Decoding runs a whole step inside one parallel region: every thread of the
# team calls these with its id, does its share, and meets the others at the
# barrier that ends each function. They reproduce the region-per-call
# kernels' partitioning and order of operations (gemv's bands and reduction,
# lm_head's chunks, one dot product per row), so results are identical.


@always_inline
def emit_rows[
    RESID: Bool, GELU: Bool, F: def(Int) -> Float32
](out_: FPtr, o0: Int, o1: Int, row: F):
    """out[o] = epilogue(row(o)) for o in [o0, o1): the values are computed
    one row at a time, but the epilogue (GELU, residual add, store) runs 16
    at a time with store_out, and only leftover rows use the scalar finish.

    GELU's tanh is ~7x faster on 16 lanes than one value at a time
    (research/test_gelu.mojo), and SIMD tanh gives exactly the scalar
    results lane by lane, so the output is unchanged bit for bit.
    """
    var o = o0
    while o + 16 <= o1:
        var v = SIMD[DType.float32, 16](0)
        comptime for k in range(16):
            v[k] = row(o + k)
        store_out[RESID, GELU, 16](out_, o, v)
        o += 16
    while o < o1:
        finish[RESID, GELU](out_, o, row(o))
        o += 1


def linear_team[
    W: WeightMatrix, //, RESID: Bool = False, GELU: Bool = False, BIAS: Bool = True
](
    tid: Int,
    team: Team,
    out_: FPtr,
    x: FPtr,
    w: W,
    b: FPtr,
    IN: Int,
    OUT: Int,
    scratch: FPtr,
    xq: I16Ptr,
    xp: I16Ptr,
    xs: FPtr,
    sx: FPtr,
):
    """One token: out[OUT] (+)= act(x[IN] @ w + b), by thread tid of the
    team; ends with a barrier.

    scratch: gemv partial sums (nt * OUT floats). xq, xp (IN int16 each), xs
    (IN / 16 floats) and sx (2 floats) are the integer path's quantized and
    permuted activations and their scale.
    """
    var nt = team.nt
    comptime if W.ACT16:
        # Thread 0 quantizes (and, for layouts that need it, reorders) the
        # activations; everyone then computes its rows.
        if tid == 0:
            quantize_rows(x, 1, IN, xq, sx)
            sx[unsafe_offset=1] = 1 if w.permute_x_i16(xq, IN, xp, xs) else 0
        team.wait()
        var x1 = xp if sx[unsafe_offset=1] != 0 else xq
        var s = sx[unsafe_offset=0]
        var r = split(OUT, tid, nt)

        def row_i16(o: Int) {imm} -> Float32:
            var v = w.dot_row_i16(o, x1, xs) * s
            comptime if BIAS:
                v += b[unsafe_offset=o]
            return v

        emit_rows[RESID, GELU](out_, r[0], r[1], row_i16)
    elif W.OUT_MAJOR:
        var r = split(OUT, tid, nt)

        def row_f32(o: Int) {imm} -> Float32:
            var v = w.dot_row(o, x)
            comptime if BIAS:
                v += b[unsafe_offset=o]
            return v

        emit_rows[RESID, GELU](out_, r[0], r[1], row_f32)
    else:
        # gemv: thread tid is part tid (the same bands as gemv with
        # nparts = nt), then the partial sums are added in the same order.
        var rows = (IN + nt - 1) // nt
        var acc = scratch.unsafe_offset(tid * OUT)
        for j in range(0, OUT, NW):
            acc.unsafe_store(j, F32V(0))
        var i = tid * rows
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
        team.wait()
        var cr = split(OUT // NW, tid, nt)
        for c in range(cr[0], cr[1]):
            var j = c * NW
            var v = F32V(0)
            comptime if BIAS:
                v = b.unsafe_load[width=NW](j)
            for pi in range(nt):
                v += scratch.unsafe_load[width=NW](pi * OUT + j)
            store_out[RESID, GELU](out_, j, v)
    team.wait()


def head_team[
    W: WeightMatrix, //
](
    tid: Int,
    team: Team,
    logits: FPtr,
    h: FPtr,
    w: W,
    V: Int,
    C: Int,
    xq: I16Ptr,
    xp: I16Ptr,
    xs: FPtr,
    sx: FPtr,
):
    """One token: logits[V] = h[C] @ w[V, C]^T, by thread tid of the team;
    ends with a barrier. Same arithmetic as `head` for T = 1."""
    comptime if W.OUT_MAJOR:
        linear_team[BIAS=False](tid, team, logits, h, w, h, C, V, h, xq, xp, xs, sx)
    else:
        # lm_head's 512-row chunks, divided among the threads.
        comptime CHUNK = 512
        var r = split((V + CHUNK - 1) // CHUNK, tid, team.nt)
        for ci in range(r[0], r[1]):
            var end = min(V, (ci + 1) * CHUNK)
            for v in range(ci * CHUNK, end):
                var d = F32V(0)
                for i in range(0, C, NW):
                    d = h.unsafe_load[width=NW](i).fma(w.load[NW](v, i), d)
                logits[unsafe_offset=v] = d.reduce_add()
        team.wait()
