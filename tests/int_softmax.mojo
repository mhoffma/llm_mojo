"""Unit test for intmath.masked_softmax.

Checks the SIMD kernel against the scalar reference (they must match
exactly) and both against float64 softmax, over attention-like scores with
and without masks, for lengths that are and aren't multiples of 16.

    uv run mojo run -I . tests/int_softmax.mojo
"""

from std.math import exp, log
from std.memory.alloc import unsafe_alloc
from std.random import random_float64, randn_float64, seed

from intmath import masked_softmax, masked_softmax_ref, exp_multiplier, MASKED


def main():
    seed(11)
    var scale = 1.0 / 65536  # scores with 16 fraction bits, as integer attention uses
    var mult = exp_multiplier(scale)
    var worst_abs = Float64(0)
    var worst_kl = Float64(0)
    var mismatches = 0
    var cases = 0
    for n in [1, 7, 16, 33, 100, 511, 1024]:
        for spread in [1.0, 4.0, 10.0]:
            for use_mask in [False, True]:
                var nr = (n + 15) // 16 * 16
                var x = unsafe_alloc[Int32](nr)
                var mask = unsafe_alloc[Int32](nr)
                var e = unsafe_alloc[Int32](nr)
                var p = unsafe_alloc[UInt16](nr)
                var pr = unsafe_alloc[UInt16](nr)
                for i in range(nr):
                    x[unsafe_offset=i] = Int32(Int(randn_float64(0, spread) / scale))
                    var masked = use_mask and i > 0 and random_float64() < 0.3
                    mask[unsafe_offset=i] = MASKED if masked else Int32(Int(randn_float64(0, 0.5) / scale)) if use_mask else 0
                if use_mask:
                    masked_softmax[True](x, mask, n, mult, e, p)
                else:
                    masked_softmax[False](x, mask, n, mult, e, p)
                masked_softmax_ref(x, mask, use_mask, n, mult, pr)
                # exact
                var m = Float64(-1e300)
                for i in range(n):
                    if not use_mask or mask[unsafe_offset=i] != MASKED:
                        var v = Float64(Int(x[unsafe_offset=i])) + (Float64(Int(mask[unsafe_offset=i])) if use_mask else 0)
                        m = max(m, v * scale)
                var z = Float64(0)
                for i in range(n):
                    if not use_mask or mask[unsafe_offset=i] != MASKED:
                        var v = Float64(Int(x[unsafe_offset=i])) + (Float64(Int(mask[unsafe_offset=i])) if use_mask else 0)
                        z += exp(v * scale - m)
                var kl = Float64(0)
                for i in range(nr):
                    if p[unsafe_offset=i] != pr[unsafe_offset=i] and i < n:
                        mismatches += 1
                    if i >= n:
                        if p[unsafe_offset=i] != 0:
                            mismatches += 1  # padding must get probability 0
                        continue
                    var want = Float64(0)
                    if not use_mask or mask[unsafe_offset=i] != MASKED:
                        var v = Float64(Int(x[unsafe_offset=i])) + (Float64(Int(mask[unsafe_offset=i])) if use_mask else 0)
                        want = exp(v * scale - m) / z
                    var got = Float64(Int(p[unsafe_offset=i])) / 32768
                    worst_abs = max(worst_abs, abs(got - want))
                    if want > 0:
                        kl += want * (log(want) - log(max(got, 0.5 / 32768)))
                worst_kl = max(worst_kl, kl)
                cases += 1
                x.unsafe_free()
                mask.unsafe_free()
                e.unsafe_free()
                p.unsafe_free()
                pr.unsafe_free()
    var ok = mismatches == 0 and worst_abs < 1e-4 and worst_kl < 2e-3
    print(
        "masked_softmax:", cases, "cases | SIMD vs scalar mismatches:", mismatches,
        "| max |p - exact|:", worst_abs, "| max KL:", worst_kl,
        "PASS" if ok else "FAIL",
    )
