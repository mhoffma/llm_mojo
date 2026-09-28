"""Probe: can GELU run in integers, and how accurately and fast?

GPT-2's MLP applies GELU in its tanh form (the model was trained with it):

    GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

In our integer (-a16) path it's still float: the int32 matmul sums become
float, GELU runs in float32, and the result is quantized again for the next
matmul. An integer pipeline would need GELU on fixed-point numbers. This
compares, on inputs and outputs in Q12 (x = q / 4096, resolution 2.4e-4):

  A  float, tanh form (what kernels.store_out / finish do now), scalar and
     SIMD: the reference arithmetic
  B  float, x * sigmoid(1.702 x): a cheaper known approximation (not
     GPT-2's exact function), SIMD
  C  integer lookup table: one entry per Q12 input in [-8, 8)
     (65536 int32 = 256 KB)
  D  integer small table + linear interpolation: 513 entries, 1/32 apart
     (2 KB, fits L1), interpolated with the input's low 7 bits
  E  integer i-GELU (I-BERT): erf approximated by a clipped second-order
     polynomial, erf(z) ~ sgn(z) [a (min(|z|, -b) + b)^2 + 1] with
     a = -0.2888, b = -1.769, and GELU = x/2 (1 + erf(x / sqrt 2)), in
     fixed point. It approximates the erf form, not the tanh form.

Outside [-8, 8) the integer versions use GELU(x) = x (x >= 8) and 0
(x < -8): exact to far below Q12's resolution there.

It reports the largest error vs GPT-2's tanh form in float64 over x in
[-12, 12], the mean error over x ~ N(0, 2), the speed on one thread, and
checks each integer method's SIMD and scalar results agree.

Run: uv run mojo run research/test_gelu.mojo

Results on the i7-1160G7, one thread, 2026-09-28:

                                  max |err|   mean |err|   ns/element
  A float tanh form, SIMD         1.0e-6      4.8e-8       0.40
  A float tanh form, scalar         (same)      (same)     3.04
  B float x*sigmoid(1.702x)       2.1e-2      8.3e-3       0.42
  C int table (256 KB)            2.5e-4      7.0e-5       1.21
  D int table + interp (2 KB)     4.0e-4      8.5e-5       1.49
  E int i-GELU (I-BERT)           1.8e-2      6.6e-3       0.62

  Integer methods: vector lanes == single-lane results.

Conclusions:
  - The tables are accurate to about one Q12 step (C's error is essentially
    the input and output rounding); D is nearly as good from a table 128x
    smaller, which stays in L1: the practical integer GELU.
  - x*sigmoid(1.702x) and I-BERT's i-GELU miss by ~2e-2, ~80x worse: both
    approximate the erf form of GELU, while GPT-2 was trained with the tanh
    form.
  - On this CPU integer GELU is slower than SIMD float tanh (0.40 ns): each
    of the 16 lanes reads the table separately, and i-GELU needs 64-bit
    multiplies. Like the integer softmax, it only pays off in a fully
    integer pipeline.
  - Side finding: decode's `finish` computes GELU with scalar tanh (3.0 ns
    vs 0.40 SIMD). At 36,864 GELUs per token that's ~0.11 ms, ~2-3% of a
    decode step at ~237 tok/s; computing the MLP up-projection's outputs
    16 at a time in the epilogue would recover most of it.
"""

from std.math import tanh, exp, sqrt
from std.memory.alloc import unsafe_alloc
from std.random import randn_float64, seed
from std.time import perf_counter_ns

comptime F = 12  # fraction bits (Q12)
comptime ONE = 1 << F
comptime LIM = 8 * ONE  # tables cover [-8, 8)
comptime L = 16
comptime I32V = SIMD[DType.int32, L]
comptime I64V = SIMD[DType.int64, L]
comptime F32V = SIMD[DType.float32, L]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]

comptime SB = 7  # interpolation table spacing: 2^7 Q12 steps = 1/32
comptime NI = (2 * LIM) >> SB  # 512 intervals, 513 entries


def gelu_ref(x: Float64) -> Float64:
    """GPT-2's GELU (tanh form) in float64: the target."""
    return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


# ===----------------------------------------------------------------------=== #
# A, B: float
# ===----------------------------------------------------------------------=== #


@always_inline
def gelu_tanh[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """A: exactly kernels.store_out's GELU."""
    comptime s = Float32(0.7978845608028654)
    return 0.5 * x * (1 + tanh(s * (x + 0.044715 * x * x * x)))


@always_inline
def gelu_sigmoid[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """B: x * sigmoid(1.702 x)."""
    return x / (1 + exp(-1.702 * x))


# ===----------------------------------------------------------------------=== #
# C, D: tables
# ===----------------------------------------------------------------------=== #


def build_direct() -> I32Ptr:
    var t = unsafe_alloc[Int32](2 * LIM)
    for i in range(2 * LIM):
        var x = Float64(i - LIM) / ONE
        t[unsafe_offset=i] = Int32(Int(round_half(gelu_ref(x) * ONE)))
    return t


def build_interp() -> I32Ptr:
    var t = unsafe_alloc[Int32](NI + 1)
    for k in range(NI + 1):
        var x = Float64((k << SB) - LIM) / ONE
        t[unsafe_offset=k] = Int32(Int(round_half(gelu_ref(x) * ONE)))
    return t


@always_inline
def round_half(x: Float64) -> Float64:
    return Float64(Int(x + 0.5)) if x >= 0 else -Float64(Int(-x + 0.5))


@always_inline
def outside(q: I32V, r: I32V) -> I32V:
    """Replaces lanes outside [-8, 8): x for x >= 8, 0 for x < -8."""
    return q.ge(LIM).select(q, q.lt(-LIM).select(I32V(0), r))


@always_inline
def gelu_direct_v(q: I32V, t: I32Ptr) -> I32V:
    """C: one table read per lane (lanes collected one at a time)."""
    var r = I32V(0)
    comptime for k in range(L):
        var i = min(max(Int(q[k]) + LIM, 0), 2 * LIM - 1)
        r[k] = t[unsafe_offset=i]
    return outside(q, r)


@always_inline
def gelu_interp_v(q: I32V, t: I32Ptr) -> I32V:
    """D: two neighbouring table entries per lane, then vector
    interpolation: r = t0 + (t1 - t0) * frac / 2^SB, rounded."""
    var u = min(max(q + LIM, I32V(0)), I32V(2 * LIM - 1))
    var idx = u >> SB
    var frac = u & ((1 << SB) - 1)
    var t0 = I32V(0)
    var t1 = I32V(0)
    comptime for k in range(L):
        var i = Int(idx[k])
        t0[k] = t[unsafe_offset=i]
        t1[k] = t[unsafe_offset = i + 1]
    var r = t0 + (((t1 - t0) * frac + (1 << (SB - 1))) >> SB)
    return outside(q, r)


# ===----------------------------------------------------------------------=== #
# E: i-GELU (I-BERT)
# ===----------------------------------------------------------------------=== #

comptime INV_SQRT2 = 11585  # round(2^14 / sqrt 2)
comptime B_ = 7246  # round(1.769 * 2^12)
comptime A_ = -4732  # round(-0.2888 * 2^14)


@always_inline
def gelu_ibert_v(q: I32V) -> I32V:
    """E: all in int64 lanes, Q12, with rounding before each shift."""
    var x = q.cast[DType.int64]()
    var z = (x * INV_SQRT2 + (1 << 13)) >> 14  # x / sqrt 2
    var az = abs(z)
    var tt = min(az, I64V(B_)) - B_  # <= 0
    var erf_abs = ((I64V(A_) * tt * tt + (1 << (14 + F - 1))) >> (14 + F)) + ONE
    var erf = z.lt(0).select(-erf_abs, erf_abs)
    var r = (x * (ONE + erf) + (1 << F)) >> (F + 1)  # x/2 (1 + erf)
    return r.cast[DType.int32]()


# ===----------------------------------------------------------------------=== #
# Measurements
# ===----------------------------------------------------------------------=== #


def to_q(x: Float64) -> Int32:
    return Int32(Int(round_half(x * ONE)))


def eval_int[method: Int](x: Float64, td: I32Ptr, ti: I32Ptr) -> Float64:
    """GELU(x) by integer method C (0), D (1) or E (2), via one lane."""
    var q = I32V(to_q(x))
    var r: I32V
    comptime if method == 0:
        r = gelu_direct_v(q, td)
    elif method == 1:
        r = gelu_interp_v(q, ti)
    else:
        r = gelu_ibert_v(q)
    return Float64(Int(r[0])) / ONE


def accuracy(td: I32Ptr, ti: I32Ptr):
    var names = ["A float tanh form (now)", "B float x*sigmoid(1.702x)", "C int table (256 KB)", "D int table+interp (2 KB)", "E int i-GELU (I-BERT)"]
    var worst = List[Float64](length=5, fill=0)
    var where = List[Float64](length=5, fill=0)
    # Sweep [-12, 12].
    var n = 240001
    for i in range(n):
        var x = -12.0 + 24.0 * Float64(i) / Float64(n - 1)
        var want = gelu_ref(x)
        var xf = Float32(x)
        var got = List[Float64](length=5, fill=0)
        got[0] = Float64(gelu_tanh[1](xf))
        got[1] = Float64(gelu_sigmoid[1](xf))
        got[2] = eval_int[0](x, td, ti)
        got[3] = eval_int[1](x, td, ti)
        got[4] = eval_int[2](x, td, ti)
        for m in range(5):
            var e = abs(got[m] - want)
            if e > worst[m]:
                worst[m] = e
                where[m] = x
    # Mean error under x ~ N(0, 2).
    var mean = List[Float64](length=5, fill=0)
    var ns = 200000
    for _ in range(ns):
        var x = randn_float64(0, 2)
        var want = gelu_ref(x)
        var xf = Float32(x)
        mean[0] += abs(Float64(gelu_tanh[1](xf)) - want)
        mean[1] += abs(Float64(gelu_sigmoid[1](xf)) - want)
        mean[2] += abs(eval_int[0](x, td, ti) - want)
        mean[3] += abs(eval_int[1](x, td, ti) - want)
        mean[4] += abs(eval_int[2](x, td, ti) - want)
    print("accuracy vs GPT-2's GELU (tanh form, float64); Q12 step = 2.4e-4:")
    print("   method                         max |err| on [-12,12] (at x)     mean |err|, x ~ N(0,2)")
    for m in range(5):
        print("  ", names[m], ":", worst[m], "(", where[m], ")   ", mean[m] / Float64(ns))


def speed(td: I32Ptr, ti: I32Ptr):
    var n = 1 << 20
    var reps = 20
    var xf = unsafe_alloc[Float32](n)
    var of = unsafe_alloc[Float32](n)
    var q = unsafe_alloc[Int32](n)
    var oq = unsafe_alloc[Int32](n)
    for i in range(n):
        var x = randn_float64(0, 2)
        xf[unsafe_offset=i] = Float32(x)
        q[unsafe_offset=i] = to_q(x)
    var check = Float64(0)

    var t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(n):
            of[unsafe_offset=i] = gelu_tanh[1](xf[unsafe_offset=i])
    var a_scalar = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(of[unsafe_offset=7])

    t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(0, n, L):
            of.unsafe_store(i, gelu_tanh[L](xf.unsafe_load[width=L](i)))
    var a_simd = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(of[unsafe_offset=7])

    t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(0, n, L):
            of.unsafe_store(i, gelu_sigmoid[L](xf.unsafe_load[width=L](i)))
    var b_simd = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(of[unsafe_offset=7])

    t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(0, n, L):
            oq.unsafe_store(i, gelu_direct_v(q.unsafe_load[width=L](i), td))
    var c = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(Int(oq[unsafe_offset=7]))

    t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(0, n, L):
            oq.unsafe_store(i, gelu_interp_v(q.unsafe_load[width=L](i), ti))
    var d = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(Int(oq[unsafe_offset=7]))

    t0 = perf_counter_ns()
    for _ in range(reps):
        for i in range(0, n, L):
            oq.unsafe_store(i, gelu_ibert_v(q.unsafe_load[width=L](i)))
    var e = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    check += Float64(Int(oq[unsafe_offset=7]))

    print("speed, one thread, ns/element (inputs ~ N(0, 2)):")
    print("   A float tanh form, scalar (as in decode's finish):", a_scalar)
    print("   A float tanh form, SIMD (as in store_out):        ", a_simd)
    print("   B float x*sigmoid(1.702x), SIMD:                  ", b_simd)
    print("   C int table (256 KB):                            ", c)
    print("   D int table + interpolation (2 KB):              ", d)
    print("   E int i-GELU (I-BERT), SIMD:                     ", e)
    if check == 12345:
        print(check)
    xf.unsafe_free()
    of.unsafe_free()
    q.unsafe_free()
    oq.unsafe_free()


def consistency(td: I32Ptr, ti: I32Ptr):
    """Each integer method gives the same result in a full 16-lane vector
    as in lane 0 alone (the lane-by-lane table reads and the vector
    arithmetic agree)."""
    var bad = 0
    for i in range(0, 1 << 16, L):
        var qv = I32V(0)
        comptime for k in range(L):
            qv[k] = Int32((i + k) * 3 - 3 * 32768)  # covers [-12, 12] in Q12
        var c = gelu_direct_v(qv, td)
        var d = gelu_interp_v(qv, ti)
        var e = gelu_ibert_v(qv)
        comptime for k in range(L):
            var one = I32V(qv[k])
            if gelu_direct_v(one, td)[0] != c[k] or gelu_interp_v(one, ti)[0] != d[k] or gelu_ibert_v(one)[0] != e[k]:
                bad += 1
    print("vector lanes == single-lane results:", "yes" if bad == 0 else "no (" + String(bad) + ")")


def main():
    seed(9)
    var td = build_direct()
    var ti = build_interp()
    accuracy(td, ti)
    consistency(td, ti)
    speed(td, ti)
    td.unsafe_free()
    ti.unsafe_free()
