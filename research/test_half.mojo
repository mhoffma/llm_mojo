"""Probe: how well does this CPU handle float16 and bfloat16 weights?

The CPU (i7-1160G7) has F16C but no AVX512_FP16 or AVX512_BF16, so neither
16-bit type can be computed with directly. The plan is to *store* weights as
16 bits and widen them to float32 in registers before each FMA. This checks:

  1. precision: how much error each format adds to GPT-2-like weights,
  2. conversion: that Mojo's `.cast` is exact and uses the right instructions
     (VCVTPH2PS for float16; a 16-bit shift for bfloat16),
  3. compute cost: how much the widening slows an FMA loop from L1 cache,
  4. the real payoff: a multi-threaded dot product streaming 256 MB of
     float32 weights from memory vs 128 MB of 16-bit weights, which is
     what one decode step does.

The two formats split their 16 bits differently:

  float32   1 sign |  8 exponent | 23 mantissa
  float16   1 sign |  5 exponent | 10 mantissa   more precision, max 65504
  bfloat16  1 sign |  8 exponent |  7 mantissa   float32's range, less precision

bfloat16 is literally the top half of a float32, which is why widening it is
just a shift.

Run:     MODULAR_THREAD_BUSY_WAIT_US=0 uv run mojo run research/test_half.mojo
Inspect: uv run mojo build --emit asm -o /tmp/half.s research/test_half.mojo
         grep -cE 'vcvtph2ps|vpslld' /tmp/half.s

Results on the i7-1160G7, 2026-09-27:

  Conversions are exact; the bfloat16 shift equals `.cast` on all 65536
  patterns. Assembly: float16 widens with VCVTPH2PS (F16C); bfloat16 with
  VPMOVZXWD + VPSLLD (zero-extend, shift) whether written by hand or not.

  Relative error on normal(0, 0.1) weights:
                mean      max
    float16     0.018%    19%   <- max comes from tiny weights (< 6e-5) that
    bfloat16    0.14%     0.39%    fall into float16's subnormal range, where
                                   precision drops; their absolute error is
                                   still tiny. float16 overflows above 65504.

  Widen + FMA from L1, one thread:  16-bit is ~11% slower than float32
  (one extra instruction per FMA).

  Streaming 64M weights from DRAM, all threads:
    float32   13.8 G weights/s   55 GB/s
    float16   25.2 G weights/s   50 GB/s   1.83x
    bfloat16  26.2 G weights/s   52 GB/s   1.90x

Conclusions: when compute-bound (prefill), 16-bit weights cost ~11%; when
memory-bound (decode), they give ~1.85x. float16 is 8x more precise than
bfloat16 for weights this size, and GPT-2's weights fit float16's range, so
float16 is the better 16-bit storage format here.
"""

from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns
from std.random import random_float64, randn_float64
from std.math import abs
from std.sys import size_of
from std.runtime import parallelism_level
from max.algorithm import parallelize

comptime W = 16  # float32 lanes in a 512-bit register
comptime F32V = SIMD[DType.float32, W]


# ===----------------------------------------------------------------------=== #
# Widening, two ways
# ===----------------------------------------------------------------------=== #


@always_inline
def widen[dt: DType](v: SIMD[dt, W]) -> F32V:
    """The library way: `.cast` to float32. Works for any source dtype."""
    return v.cast[DType.float32]()


@always_inline
def bf16_by_shift(v: SIMD[DType.bfloat16, W]) -> F32V:
    """The by-hand way for bfloat16: put its 16 bits in the top of a float32.

    `bitcast` reinterprets bits without changing them (free), and the shift
    moves each 16-bit value into the upper half of a 32-bit lane.
    """
    var bits = bitcast[DType.uint16, W](v).cast[DType.uint32]() << 16
    return bitcast[DType.float32, W](bits)


# ===----------------------------------------------------------------------=== #
# 1 + 2. Precision and conversion correctness
# ===----------------------------------------------------------------------=== #


def precision():
    """Rounds a million GPT-2-sized weights to each format and back.

    GPT-2's weights are roughly normal with standard deviation ~0.02-0.2, so
    that's what we draw. Reports the worst and average relative error.
    """
    var n = 1_000_000
    var max16 = Float64(0)
    var maxbf = Float64(0)
    var sum16 = Float64(0)
    var sumbf = Float64(0)
    for _ in range(n):
        var x = Float32(randn_float64(0, 0.1))
        if x == 0:
            continue
        var r16 = x.cast[DType.float16]().cast[DType.float32]()
        var rbf = x.cast[DType.bfloat16]().cast[DType.float32]()
        var e16 = Float64(abs((r16 - x) / x))
        var ebf = Float64(abs((rbf - x) / x))
        max16 = max(max16, e16)
        maxbf = max(maxbf, ebf)
        sum16 += e16
        sumbf += ebf
    print("precision on 1M normal(0, 0.1) weights, relative error:")
    print("   float16  max", max16, " mean", sum16 / Float64(n))
    print("   bfloat16 max", maxbf, " mean", sumbf / Float64(n))
    print("   (theory: float16 max 2^-11 =", 1.0 / 2048.0,
          ", bfloat16 max 2^-8 =", 1.0 / 256.0, ")")

    # Weights this small also sit near float16's lower limit: below ~6e-5
    # float16 loses precision ("subnormals"), and below ~6e-8 it rounds to 0.
    var tiny = Float32(1e-6)
    print("   1e-6 as float16 ->", tiny.cast[DType.float16]().cast[DType.float32](),
          " as bfloat16 ->", tiny.cast[DType.bfloat16]().cast[DType.float32]())
    var big = Float32(70000)
    print("   70000 as float16 ->", big.cast[DType.float16]().cast[DType.float32](),
          " as bfloat16 ->", big.cast[DType.bfloat16]().cast[DType.float32]())


def conversion_checks() -> Bool:
    """Checks known values, and that the bfloat16 shift matches `.cast`."""
    var ok = True
    # 1/3 rounded to nearest: float16 0x3555, bfloat16 0x3EAB.
    var third = Float32(1.0) / 3
    var h = third.cast[DType.float16]().cast[DType.float32]()
    var b = third.cast[DType.bfloat16]().cast[DType.float32]()
    print("1/3 -> float16", h, "(want 0.33325195), bfloat16", b,
          "(want 0.33398438)")
    ok = ok and h == Float32(0.333251953125) and b == Float32(0.333984375)

    # The shift must agree with .cast for every one of the 65536 bit patterns
    # (NaNs excluded, since NaN != NaN).
    for bits in range(65536):
        var v = SIMD[DType.uint16, W](UInt16(bits))
        var x = bitcast[DType.bfloat16, W](v)
        var a = widen(x)
        var c = bf16_by_shift(x)
        if a[0] == a[0] and a != c:
            ok = False
    return ok


# ===----------------------------------------------------------------------=== #
# 3. Compute cost: FMA loop over an L1-resident buffer
# ===----------------------------------------------------------------------=== #

comptime NACC = 8  # independent accumulators, as in test_vnni.mojo
comptime NBUF = 64  # vectors in the buffer: 2-4 KB, stays in L1
comptime ITERS = 20_000_000


def l1_fma[dt: DType, SHIFT: Bool = False](name: StaticString) -> Float64:
    """Loads a dt vector, widens it to float32, and FMAs it into NACC sums."""
    var buf = unsafe_alloc[Scalar[dt]](NBUF * W)
    for i in range(NBUF * W):
        buf[unsafe_offset=i] = Float32(random_float64(-1, 1)).cast[dt]()
    var x = Array[F32V, length=NACC](fill=F32V(0))
    comptime for k in range(NACC):
        x[k] = F32V(Float32(k) * 0.01)
    var acc = Array[F32V, length=NACC](fill=F32V(0))
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        var raw = buf.unsafe_load[width=W]((it % NBUF) * W)
        var w: F32V
        comptime if SHIFT:
            w = bf16_by_shift(rebind[SIMD[DType.bfloat16, W]](raw))
        else:
            w = widen(raw)
        comptime for k in range(NACC):
            acc[k] = w.fma(x[k], acc[k])
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var total = F32V(0)
    comptime for k in range(NACC):
        total += acc[k]
    buf.unsafe_free()
    var gmacs = Float64(ITERS * NACC * W) / secs / 1e9
    print("  ", name, ":", gmacs, "GMAC/s  (checksum", total.reduce_add(), ")")
    return gmacs


# ===----------------------------------------------------------------------=== #
# 4. The payoff: streaming weights from memory, all threads
# ===----------------------------------------------------------------------=== #

comptime NSTREAM = 64 * 1024 * 1024  # 64M weights: 256 MB f32, 128 MB 16-bit


def stream_dot[dt: DType](name: StaticString) -> Float64:
    """Dot product of 64M weights with an L1-resident activation vector.

    This is the shape of a decode step: every weight is read once from DRAM
    and used for one FMA. Returns billions of weights per second; the
    activation vector is small and repeats, as it would across output rows.
    """
    var wts = unsafe_alloc[Scalar[dt]](NSTREAM)
    for i in range(0, NSTREAM, W):
        wts.unsafe_store(i, SIMD[dt, W](Scalar[dt](0.01)))
    var act = unsafe_alloc[Float32](1024)
    for i in range(1024):
        act[unsafe_offset=i] = 1
    var nt = parallelism_level()
    var partial = unsafe_alloc[Float32](nt)
    var best = Float64(0)
    for _ in range(3):  # best of 3: the first pass also warms the page tables
        var t0 = perf_counter_ns()

        def part(t: Int) {imm}:
            var chunk = NSTREAM // nt
            var a0 = F32V(0)
            var a1 = F32V(0)
            var i = t * chunk
            while i < (t + 1) * chunk:
                var w0 = wts.unsafe_load[width=W](i).cast[DType.float32]()
                var w1 = wts.unsafe_load[width=W](i + W).cast[DType.float32]()
                a0 = w0.fma(act.unsafe_load[width=W](i % 1024), a0)
                a1 = w1.fma(act.unsafe_load[width=W]((i + W) % 1024), a1)
                i += 2 * W
            partial[unsafe_offset=t] = (a0 + a1).reduce_add()

        parallelize(part, nt)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        best = max(best, Float64(NSTREAM) / secs / 1e9)
    var gbps = best * Float64(size_of[Scalar[dt]]())
    print("  ", name, ":", best, "G weights/s (", gbps, "GB/s )  check",
          partial[unsafe_offset=0])
    wts.unsafe_free()
    act.unsafe_free()
    partial.unsafe_free()
    return best


def main():
    print("conversion checks:", "PASS" if conversion_checks() else "FAIL")
    precision()

    print("widen + FMA, one thread, weights from L1 cache:")
    var f32 = l1_fma[DType.float32]("float32         ")
    var f16 = l1_fma[DType.float16]("float16 (.cast) ")
    var bfc = l1_fma[DType.bfloat16]("bfloat16 (.cast)")
    var bfs = l1_fma[DType.bfloat16, SHIFT=True]("bfloat16 (shift)")
    print("   relative to float32: f16", f16 / f32, "| bf16 cast", bfc / f32,
          "| bf16 shift", bfs / f32)

    print("streaming dot product, all threads, weights from DRAM:")
    var s32 = stream_dot[DType.float32]("float32 ")
    var s16 = stream_dot[DType.float16]("float16 ")
    var sbf = stream_dot[DType.bfloat16]("bfloat16")
    print("   weights/s relative to float32: f16", s16 / s32, "| bf16", sbf / s32)
