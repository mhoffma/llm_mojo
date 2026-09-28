"""Integer math for integer attention: a masked softmax in fixed point.

`masked_softmax` computes softmax(x + mask) with integer arithmetic only
(research/test_softmax.mojo is the probe that developed it):

  1. Subtract the max:   softmax(x)_i = e^(x_i - m) / sum_j e^(x_j - m).
  2. Change the base:    e^d = 2^(d * log2(e)); powers of 2 are shifts.
  3. Split the exponent: y = -d * log2(e) = q + r (integer q, fraction r),
                         so 2^(-y) = 2^(-r) >> q, with 2^(-r) on [0, 1) from a
                         degree-3 polynomial (max relative error 1.4e-4),
                         exact at both ends.
  4. Normalize with one division: p_i = e_i * (2^61 / sum) >> 46, rounded.

Formats:
  x, mask   int32 scores in a shared scale s; mask entries are added, and
            MASKED (Int32.MIN) means minus infinity (probability exactly 0).
            Positions at or after n are masked too, so a caller can pass a
            buffer padded to a multiple of 16.
  mult      exp_multiplier(s): M such that ((m - x) * M) >> MB is the base-2
            exponent with FB fraction bits. The one floating-point step,
            done once per scale (a constant for a fixed s).
  e         scratch, Q30 values of 2^(-y)
  probs     uint16 in Q15 (probabilities sum to ~32768), rounded
"""

from std.memory.alloc import unsafe_alloc

comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]

comptime FB = 16  # fraction bits of the base-2 exponent
comptime MB = 24  # extra fraction bits of the multiplier M
comptime MASKED = Int32.MIN  # mask sentinel: minus infinity
comptime LOG2E = 1.4426950408889634
comptime D_CAP = 1 << 31  # score differences are capped here, so d * M fits in 64 bits
comptime SM_LANES = 16

# 2^(-r) for r in [0, 1), in Q30: 1 - 0.6915983741 r + 0.2311609834 r^2
# - 0.0395626093 r^3. The coefficients sum to exactly 2^29 (0.5).
comptime C0 = 1073741824
comptime C1 = -742598100
comptime C2 = 248207216
comptime C3 = -42480028

comptime I64V = SIMD[DType.int64, SM_LANES]


def exp_multiplier(scale: Float64) -> Int:
    """M for scores in scale `scale` (score = scale * x)."""
    return Int(scale * LOG2E * Float64(1 << (FB + MB)) + 0.5)


@always_inline
def exp2_neg(d: I64V, mult: I64V) -> I64V:
    """2^(-d * scale * log2 e) in Q30 for score differences d >= 0: the
    integer part of the exponent as a shift, the fraction by polynomial.
    Underflow (exponent >= 31) gives 0."""
    var y = (min(d, I64V(D_CAP)) * mult) >> MB
    var q = y >> FB
    var r = y & ((1 << FB) - 1)
    var p = I64V(C3)
    p = ((p * r) >> FB) + I64V(C2)
    p = ((p * r) >> FB) + I64V(C1)
    p = ((p * r) >> FB) + I64V(C0)
    return q.ge(31).select(I64V(0), p >> min(q, I64V(63)))


def lane_index() -> I64V:
    """(0, 1, ..., 15)."""
    var v = I64V(0)
    for k in range(SM_LANES):
        v[k] = Int64(k)
    return v


@always_inline
def load_masked[
    HAS_MASK: Bool
](x: I32Ptr, mask: I32Ptr, i: Int, n: Int, lanes: I64V) -> Tuple[
    I64V, SIMD[DType.bool, SM_LANES]
]:
    """x + mask for lanes i .. i+15 as int64, and which lanes are masked (by
    the mask, or at or after n). lanes is lane_index()."""
    var v = x.unsafe_load[width=SM_LANES](i).cast[DType.int64]()
    var off = (lanes + I64V(i)).ge(I64V(n))
    comptime if HAS_MASK:
        var mk = mask.unsafe_load[width=SM_LANES](i)
        off = off | mk.eq(MASKED)
        v += mk.cast[DType.int64]()
    return (v, off)


def masked_exp[
    HAS_MASK: Bool
](x: I32Ptr, mask: I32Ptr, n: Int, mult: Int, e: I32Ptr) -> Int:
    """The unnormalized half of softmax: e_i = 2^((x_i + mask_i - m) log2 e)
    in Q30, for positions 0 .. n-1, where m is the max. The largest e is
    exactly 2^30 (1.0); masked positions and positions >= n get 0. Returns
    the int64 sum of the e_i.

    x, mask and e must have room for n rounded up to a multiple of 16. With
    HAS_MASK=False, mask is not read.

    Integer attention uses this directly and divides by the sum once at the
    end (out = sum e_i v_i / sum e_i), which keeps its weights at full
    precision when attention is spread over many positions.
    """
    var nr = (n + SM_LANES - 1) // SM_LANES * SM_LANES
    var lanes = lane_index()
    var neg_inf = I64V(Int.MIN)
    var mv = neg_inf
    for i in range(0, nr, SM_LANES):
        var vm = load_masked[HAS_MASK](x, mask, i, n, lanes)
        mv = max(mv, vm[1].select(neg_inf, vm[0]))
    var m = I64V(mv.reduce_max())
    var mm = I64V(mult)
    var total = I64V(0)
    for i in range(0, nr, SM_LANES):
        var vm = load_masked[HAS_MASK](x, mask, i, n, lanes)
        var ev = vm[1].select(I64V(0), exp2_neg(m - vm[0], mm))
        e.unsafe_store(i, ev.cast[DType.int32]())
        total += ev
    return Int(total.reduce_add())


def masked_softmax[
    HAS_MASK: Bool
](x: I32Ptr, mask: I32Ptr, n: Int, mult: Int, e: I32Ptr, probs: U16Ptr):
    """probs = softmax(x + mask) over positions 0 .. n-1, in Q15.

    x, mask, e and probs must have room for n rounded up to a multiple of
    16; entries from n on are ignored (their probability is written as 0).
    With HAS_MASK=False, mask is not read (positions >= n are still masked).
    """
    var nr = (n + SM_LANES - 1) // SM_LANES * SM_LANES
    var t = masked_exp[HAS_MASK](x, mask, n, mult, e)
    var inv = I64V((1 << 61) // t if t > 0 else 0)
    for i in range(0, nr, SM_LANES):
        var p = (e.unsafe_load[width=SM_LANES](i).cast[DType.int64]() * inv + (1 << 45)) >> 46
        probs.unsafe_store(i, p.cast[DType.uint16]())


def masked_softmax_ref(
    x: I32Ptr, mask: I32Ptr, has_mask: Bool, n: Int, mult: Int, probs: U16Ptr
):
    """The same computation one element at a time, for tests."""
    var m = Int.MIN
    for i in range(n):
        if not has_mask or mask[unsafe_offset=i] != MASKED:
            var v = Int(x[unsafe_offset=i])
            if has_mask:
                v += Int(mask[unsafe_offset=i])
            m = max(m, v)
    var e = List[Int](length=n, fill=0)
    var total = 0
    for i in range(n):
        if not has_mask or mask[unsafe_offset=i] != MASKED:
            var v = Int(x[unsafe_offset=i])
            if has_mask:
                v += Int(mask[unsafe_offset=i])
            var y = (min(m - v, D_CAP) * mult) >> MB
            var q = y >> FB
            var r = y & ((1 << FB) - 1)
            var p = C3
            p = ((p * r) >> FB) + C2
            p = ((p * r) >> FB) + C1
            p = ((p * r) >> FB) + C0
            e[i] = 0 if q >= 31 else p >> q
            total += e[i]
    var inv = (1 << 61) // total if total > 0 else 0
    for i in range(n):
        probs[unsafe_offset=i] = UInt16((e[i] * inv + (1 << 45)) >> 46)
