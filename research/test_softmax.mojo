"""Probe: an integer masked softmax, softmax(x + mask), without floating point.

For integer attention the scores arrive as int32 in one shared scale, and we
want probabilities as integers too. Softmax needs e^x; the integer approach
(as in I-BERT) rests on four facts:

  1. Subtract the max:  softmax(x)_i = e^(x_i - m) / sum_j e^(x_j - m), so
     every exponent is <= 0.
  2. Change the base:   e^d = 2^(d * log2(e)),  log2(e) = 1/ln(2) = 1.4427...
     Powers of 2 are cheap in integers: a shift.
  3. Split the exponent: with y = -d * log2(e) >= 0 and y = q + r (integer q,
     fraction 0 <= r < 1),  2^(-y) = 2^(-r) >> q.  Only 2^(-r), a value in
     (0.5, 1], needs approximating, over the fixed range r in [0, 1).
  4. Normalize with one division: inv = 2^K / sum, then p_i = (e_i * inv) >> K'.

The mask is additive, as in softmax(act + mask): entries are added to the
scores, and the sentinel MASKED (Int32.MIN) means minus infinity: that
position gets probability exactly 0 and is left out of the max and the sum.

Fixed-point formats:

  x, mask   int32 scores in a shared scale `s` (score = s * x)
  M         int64 multiplier = round(s * log2(e) * 2^(FB + MB)): turns a
            score difference d into a base-2 exponent with FB = 16 fraction
            bits, as (d * M) >> MB. The MB = 24 extra bits matter: a first
            version used round(s * log2(e) * 2^FB), which for s = 1/4096 is
            23.08 rounded to 23, a 0.35% error in the slope of every
            exponent, ~7% in probability at a score difference of 20.
  e         2^(-r) >> q in Q30 (1.0 = 2^30), via a degree-2 or degree-3
            polynomial fitted with its endpoints exact (so e is continuous
            across integer steps of the exponent):
              deg 2: 1 - 0.6718170695 r + 0.1718170695 r^2       (max rel err 3.2e-3)
              deg 3: 1 - 0.6915983741 r + 0.2311609834 r^2
                       - 0.0395626093 r^3                         (max rel err 1.4e-4)
  sum       int64 (up to n * 2^30)
  p         uint16 in Q15: probabilities that sum to ~32768

The probe checks a scalar reference against a 16-lane SIMD version (they
must match exactly), measures the error against float64 softmax on
attention-like scores with and without masks, and times it against float
softmax.

Run: uv run mojo run research/test_softmax.mojo

Results on the i7-1160G7, one thread, 2026-09-28 (240 cases: n = 16 to
1024, score spreads 1 to 8, with and without random and causal masks):

                           degree 2      degree 3
  max |p - exact|          6.8e-4        3.4e-5   (about 1 Q15 step)
  L1 error                 8.4e-3        7.9e-3
  KL(exact || int)         5.4e-4        5.7e-4
  mass lost below Q15      0.13%         0.15%
  SIMD == scalar           yes           yes

  speed (n = 1024):  float, scalar exp 5.8 ns/element | float, SIMD exp
  0.40 | integer, scalar 3.3 | integer, SIMD 0.88

Lessons:
  - Precision of the multiplier M matters most: without the extra MB bits,
    M rounded to 23 instead of 23.08 and errors were 100x larger (max 3.4e-3).
  - Round, don't truncate, the final shift: KL 0.016 -> 0.00057.
  - With degree 3 the remaining error is the Q15 output (probabilities
    below 1/32768 become 0 over long contexts), not the exponential.
  - On this CPU the integer softmax is accurate but slower than a SIMD float
    softmax: its 64-bit fixed-point multiplies are expensive. Both are far
    faster than the scalar-exp float softmax the attention kernel uses now.
    The integer version pays off only in an all-integer attention (int16
    probabilities times int16 values with VPDPWSSD), which also needs the
    scores in one shared scale.
"""

from std.math import exp, log
from std.memory.alloc import unsafe_alloc
from std.random import random_float64, randn_float64, seed
from std.time import perf_counter_ns

comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]

comptime FB = 16  # fraction bits of the base-2 exponent
comptime MB = 24  # extra fraction bits of the multiplier M, shifted off after multiplying
comptime Q = 30  # e values are Q30: 1.0 = 2^30
comptime MASKED = Int32.MIN  # mask sentinel: minus infinity
comptime LOG2E = 1.4426950408889634

# 2^(-r) coefficients, in Q30. Each set sums to exactly 2^29 (0.5), so the
# approximation is exact at both ends of [0, 1) in fixed point too.
comptime D2_0 = 1073741824
comptime D2_1 = -721358086
comptime D2_2 = 184487174
comptime D3_0 = 1073741824
comptime D3_1 = -742598100
comptime D3_2 = 248207216
comptime D3_3 = -42480028


def exp_multiplier(scale: Float64) -> Int:
    """M such that (d * M) >> MB is the base-2 exponent, with FB fraction
    bits, of a score difference d (in the scores' scale). Computed once per
    call; the only floating-point step, and one a real kernel would
    precompute."""
    return Int(scale * LOG2E * Float64(1 << (FB + MB)) + 0.5)


comptime D_CAP = 1 << 31
"""Score differences are capped here before multiplying, so d * M fits in
64 bits; any cap this large already gives an exponent far beyond 2^-31."""


# ===----------------------------------------------------------------------=== #
# Scalar reference
# ===----------------------------------------------------------------------=== #


@always_inline
def exp2_neg_frac[DEG: Int](r: Int) -> Int:
    """2^(-r / 2^FB) in Q30, for 0 <= r < 2^FB, by Horner's rule in fixed
    point (arithmetic right shifts keep the sign of negative terms)."""
    comptime if DEG == 2:
        var p = D2_2
        p = ((p * r) >> FB) + D2_1
        return ((p * r) >> FB) + D2_0
    else:
        var p = D3_3
        p = ((p * r) >> FB) + D3_2
        p = ((p * r) >> FB) + D3_1
        return ((p * r) >> FB) + D3_0


def softmax_ref[
    DEG: Int
](x: I32Ptr, mask: I32Ptr, n: Int, mult: Int, e: I32Ptr, probs: U16Ptr):
    """Masked softmax, one element at a time. e is scratch (n values)."""
    var m = Int.MIN
    for i in range(n):
        if mask[unsafe_offset=i] != MASKED:
            m = max(m, Int(x[unsafe_offset=i]) + Int(mask[unsafe_offset=i]))
    var total = 0
    for i in range(n):
        var v = 0
        if mask[unsafe_offset=i] != MASKED:
            var d = m - (Int(x[unsafe_offset=i]) + Int(mask[unsafe_offset=i]))
            var y = (min(d, D_CAP) * mult) >> MB  # base-2 exponent, FB fraction bits
            var q = y >> FB
            if q < 31:
                v = exp2_neg_frac[DEG](y & ((1 << FB) - 1)) >> q
        e[unsafe_offset=i] = Int32(v)
        total += v
    var inv = (1 << 61) // total if total > 0 else 0
    for i in range(n):
        # Round to nearest (+ half a step) rather than truncate.
        probs[unsafe_offset=i] = UInt16(((Int(e[unsafe_offset=i]) * inv) + (1 << 45)) >> 46)


# ===----------------------------------------------------------------------=== #
# SIMD version (16 lanes)
# ===----------------------------------------------------------------------=== #

comptime L = 16
comptime I64V = SIMD[DType.int64, L]


@always_inline
def exp2_neg_frac_v[DEG: Int](r: I64V) -> I64V:
    comptime if DEG == 2:
        var p = I64V(D2_2)
        p = ((p * r) >> FB) + I64V(D2_1)
        return ((p * r) >> FB) + I64V(D2_0)
    else:
        var p = I64V(D3_3)
        p = ((p * r) >> FB) + I64V(D3_2)
        p = ((p * r) >> FB) + I64V(D3_1)
        return ((p * r) >> FB) + I64V(D3_0)


def softmax_simd[
    DEG: Int
](x: I32Ptr, mask: I32Ptr, n: Int, mult: Int, e: I32Ptr, probs: U16Ptr):
    """The same computation, 16 elements at a time (n a multiple of 16).
    Masked lanes and lanes whose exponent shifts everything out become 0
    with a select, not a branch."""
    var neg_inf = I64V(Int.MIN)
    var mv = neg_inf
    for i in range(0, n, L):
        var mk = mask.unsafe_load[width=L](i)
        var v = x.unsafe_load[width=L](i).cast[DType.int64]() + mk.cast[DType.int64]()
        mv = max(mv, mk.eq(MASKED).select(neg_inf, v))
    var m = I64V(mv.reduce_max())
    var total = I64V(0)
    for i in range(0, n, L):
        var mk = mask.unsafe_load[width=L](i)
        var v = x.unsafe_load[width=L](i).cast[DType.int64]() + mk.cast[DType.int64]()
        var y = (min(m - v, I64V(D_CAP)) * I64V(mult)) >> MB
        var q = y >> FB
        var ev = exp2_neg_frac_v[DEG](y & ((1 << FB) - 1)) >> min(q, I64V(63))
        ev = (mk.eq(MASKED) | q.ge(31)).select(I64V(0), ev)
        e.unsafe_store(i, ev.cast[DType.int32]())
        total += ev
    var t = total.reduce_add()
    var inv = I64V((1 << 61) // t if t > 0 else 0)
    for i in range(0, n, L):
        var p = (e.unsafe_load[width=L](i).cast[DType.int64]() * inv + (1 << 45)) >> 46
        probs.unsafe_store(i, p.cast[DType.uint16]())


# ===----------------------------------------------------------------------=== #
# Float softmax, as the attention kernel does it now
# ===----------------------------------------------------------------------=== #


def softmax_float(s: FPtr, n: Int, probs: FPtr):
    var m = s[unsafe_offset=0]
    for i in range(1, n):
        m = max(m, s[unsafe_offset=i])
    var total = Float32(0)
    for i in range(n):
        var v = exp(s[unsafe_offset=i] - m)
        probs[unsafe_offset=i] = v
        total += v
    var inv = 1 / total
    for i in range(n):
        probs[unsafe_offset=i] = probs[unsafe_offset=i] * inv


comptime F32x16 = SIMD[DType.float32, L]


def softmax_float_simd(s: FPtr, n: Int, probs: FPtr):
    """Float softmax with 16-lane SIMD exp: the fair float baseline."""
    var mv = F32x16(Float32.MIN)
    for i in range(0, n, L):
        mv = max(mv, s.unsafe_load[width=L](i))
    var m = F32x16(mv.reduce_max())
    var total = F32x16(0)
    for i in range(0, n, L):
        var v = exp(s.unsafe_load[width=L](i) - m)
        probs.unsafe_store(i, v)
        total += v
    var inv = F32x16(1 / total.reduce_add())
    for i in range(0, n, L):
        probs.unsafe_store(i, probs.unsafe_load[width=L](i) * inv)


# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


struct Case(Movable):
    var x: I32Ptr
    var mask: I32Ptr
    var n: Int
    var scale: Float64

    def __init__(out self, n: Int, spread: Float64, masked_frac: Float64, causal: Bool):
        """Attention-like scores: normal with std `spread` (in score units),
        plus a few large ones; some positions masked."""
        self.n = n
        self.scale = 1.0 / 4096  # scores as int32 with 12 fraction bits
        self.x = unsafe_alloc[Int32](n)
        self.mask = unsafe_alloc[Int32](n)
        for i in range(n):
            var s = randn_float64(0, spread)
            if random_float64() < 0.02:
                s += spread * 3  # a few strongly attended positions
            self.x[unsafe_offset=i] = Int32(Int(s / self.scale))
            var masked = random_float64() < masked_frac
            if causal and i >= n * 3 // 4:  # the last quarter is "future"
                masked = True
            if i == 0:
                masked = False  # at least one position survives
            self.mask[unsafe_offset=i] = MASKED if masked else 0

    def __deinit__(deinit self):
        self.x.unsafe_free()
        self.mask.unsafe_free()


def errors[
    DEG: Int
](
    c: Case,
    mut worst_abs: Float64,
    mut worst_kl: Float64,
    mut worst_l1: Float64,
    mut worst_lost: Float64,
    mut mismatch: Bool,
):
    var e = unsafe_alloc[Int32](c.n)
    var p_ref = unsafe_alloc[UInt16](c.n)
    var p_simd = unsafe_alloc[UInt16](c.n)
    var mult = exp_multiplier(c.scale)
    softmax_ref[DEG](c.x, c.mask, c.n, mult, e, p_ref)
    softmax_simd[DEG](c.x, c.mask, c.n, mult, e, p_simd)
    # Exact softmax in float64 on the real-valued scores.
    var m = Float64(-1e300)
    for i in range(c.n):
        if c.mask[unsafe_offset=i] != MASKED:
            m = max(m, Float64(Int(c.x[unsafe_offset=i])) * c.scale)
    var z = Float64(0)
    for i in range(c.n):
        if c.mask[unsafe_offset=i] != MASKED:
            z += exp(Float64(Int(c.x[unsafe_offset=i])) * c.scale - m)
    var kl = Float64(0)
    var l1 = Float64(0)
    var lost = Float64(0)  # exact probability of positions that became 0
    for i in range(c.n):
        if p_ref[unsafe_offset=i] != p_simd[unsafe_offset=i]:
            mismatch = True
        var want = Float64(0)
        if c.mask[unsafe_offset=i] != MASKED:
            want = exp(Float64(Int(c.x[unsafe_offset=i])) * c.scale - m) / z
        var got = Float64(Int(p_ref[unsafe_offset=i])) / 32768
        worst_abs = max(worst_abs, abs(got - want))
        l1 += abs(got - want)
        if want > 0:
            if got == 0:
                lost += want
            # Probabilities below one Q15 step become 0; compare them at
            # half a step rather than as 0 (log 0).
            kl += want * (log(want) - log(max(got, 0.5 / 32768)))
    worst_kl = max(worst_kl, kl)
    worst_l1 = max(worst_l1, l1)
    worst_lost = max(worst_lost, lost)
    e.unsafe_free()
    p_ref.unsafe_free()
    p_simd.unsafe_free()


def accuracy[DEG: Int]():
    var worst_abs = Float64(0)
    var worst_kl = Float64(0)
    var worst_l1 = Float64(0)
    var worst_lost = Float64(0)
    var mismatch = False
    var cases = 0
    for n in [16, 128, 512, 1024]:
        for spread in [1.0, 3.0, 8.0]:
            for mf in [0.0, 0.3]:
                for causal in [False, True]:
                    for _ in range(5):
                        var c = Case(n, spread, mf, causal)
                        errors[DEG](
                            c, worst_abs, worst_kl, worst_l1, worst_lost, mismatch
                        )
                        cases += 1
    print("  degree", DEG, ":", cases, "cases; worst case of each measure:")
    print("     max |p - exact| over positions:     ", worst_abs)
    print("     sum |p - exact| (L1) over positions:", worst_l1)
    print("     KL(exact || int), in nats:          ", worst_kl)
    print("     probability mass below Q15 (-> 0):  ", worst_lost)
    print("     SIMD == scalar reference:           ", "no" if mismatch else "yes")


def speed():
    var n = 1024
    var reps = 20000
    var c = Case(n, 3.0, 0.0, False)
    var e = unsafe_alloc[Int32](n)
    var p = unsafe_alloc[UInt16](n)
    var s = unsafe_alloc[Float32](n)
    var pf = unsafe_alloc[Float32](n)
    for i in range(n):
        s[unsafe_offset=i] = Float32(Float64(Int(c.x[unsafe_offset=i])) * c.scale)
    var mult = exp_multiplier(c.scale)
    var check = 0

    var t0 = perf_counter_ns()
    for _ in range(reps):
        softmax_float(s, n, pf)
    var t_float = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    t0 = perf_counter_ns()
    for _ in range(reps):
        softmax_float_simd(s, n, pf)
    var t_float_simd = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    t0 = perf_counter_ns()
    for _ in range(reps):
        softmax_ref[3](c.x, c.mask, n, mult, e, p)
        check += Int(p[unsafe_offset=0])
    var t_ref = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    t0 = perf_counter_ns()
    for _ in range(reps):
        softmax_simd[3](c.x, c.mask, n, mult, e, p)
        check += Int(p[unsafe_offset=0])
    var t_simd = Float64(perf_counter_ns() - t0) / Float64(reps * n)
    print("  float softmax (scalar exp, as in attention now):", t_float, "ns/element")
    print("  float softmax, 16-lane SIMD exp:               ", t_float_simd, "ns/element")
    print("  integer, scalar reference (degree 3):          ", t_ref, "ns/element")
    print("  integer, 16-lane SIMD (degree 3):              ", t_simd, "ns/element")
    if check == 1:
        print(check, pf[unsafe_offset=0])
    e.unsafe_free()
    p.unsafe_free()
    s.unsafe_free()
    pf.unsafe_free()


def main():
    seed(3)
    print("accuracy vs float64 softmax (probabilities in Q15: 1/32768 = 3.1e-5):")
    accuracy[2]()
    accuracy[3]()
    print("speed, n = 1024, one thread:")
    speed()
