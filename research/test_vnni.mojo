"""Probe: can Mojo 1.1 use AVX-512 VNNI for int16 dot products?

The W4A16 plan multiplies int16 activations by int4 weights (unpacked to
int16) and accumulates in int32. On x86 there are two instructions for that:

  VPMADDWD  (AVX-512BW)    r[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
  VPDPWSSD  (AVX-512 VNNI) acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]

VPDPWSSD fuses the multiply, pair-add, and accumulate into one instruction.
This program checks three routes to it:

  1. calling the LLVM intrinsic directly with `llvm_intrinsic`,
  2. the VPMADDWD intrinsic plus a separate add (the fallback),
  3. plain Mojo SIMD code (widen, multiply, add) to see what LLVM does alone,

then compares their speed with float32 FMA, the kernel we use today.

Run:     uv run mojo run research/test_vnni.mojo
Inspect: uv run mojo build --emit asm -o /tmp/vnni.s research/test_vnni.mojo
         grep -c vpdpwssd /tmp/vnni.s

Results on the i7-1160G7 (Tiger Lake), one thread, 2026-09-27:

  f32 fma   ~35 GMAC/s   16 MACs per instruction (one 512-bit FMA unit)
  vpdpwssd  ~92 GMAC/s   32 MACs per instruction -> ~2.6x f32
  pmaddwd   ~92 GMAC/s   LLVM fused pmaddwd + add into vpdpwssd by itself
  portable  ~20 GMAC/s   not recognized: widening multiplies + shuffles

Conclusions: all three routes are exact; call the VNNI intrinsic directly;
the int16 kernel can do ~2.6x the multiply-adds of the f32 kernel.
"""

from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic
from std.memory import bitcast
from std.time import perf_counter_ns
from std.random import random_si64
from std.memory.alloc import unsafe_alloc

# One 512-bit register holds 32 int16s or 16 int32s.
comptime I16x32 = SIMD[DType.int16, 32]
comptime I32x16 = SIMD[DType.int32, 16]
comptime F32x16 = SIMD[DType.float32, 16]


# ===----------------------------------------------------------------------=== #
# The three routes
# ===----------------------------------------------------------------------=== #


@always_inline
def dpwssd(acc: I32x16, a: I16x32, b: I16x32) -> I32x16:
    """Route 1: VPDPWSSD via its LLVM intrinsic.

    `llvm_intrinsic[name, ReturnType](args...)` emits a call to any LLVM
    intrinsic. The argument types must match LLVM's declaration exactly. Older
    LLVMs declared the int16 operands as <16 x i32> (pairs packed in 32-bit
    lanes); the LLVM in Mojo 1.1 takes <32 x i16>, which is what we have, and
    the compiler's error message tells you which one it expects.
    `has_side_effect=False` tells the compiler the call is pure, so it may
    reorder or remove it like ordinary arithmetic.
    """
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.512", I32x16, has_side_effect=False
    ](acc, a, b)


@always_inline
def pmaddwd(acc: I32x16, a: I16x32, b: I16x32) -> I32x16:
    """Route 2: VPMADDWD (multiply and add adjacent pairs), then add."""
    return acc + llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", I32x16, has_side_effect=False
    ](a, b)


@always_inline
def portable(acc: I32x16, a: I16x32, b: I16x32) -> I32x16:
    """Route 3: plain SIMD code. Widen to int32, multiply, add the pairs.

    `deinterleave` splits even and odd lanes: (a0, a2, ...), (a1, a3, ...).
    Works on any CPU; the question is whether LLVM turns it into VNNI.
    """
    var p = a.cast[DType.int32]() * b.cast[DType.int32]()
    var halves = p.deinterleave()
    return acc + halves[0] + halves[1]


# ===----------------------------------------------------------------------=== #
# Correctness
# ===----------------------------------------------------------------------=== #


def random_i16(lo: Int, hi: Int) -> I16x32:
    var v = I16x32(0)
    for i in range(32):
        v[i] = Int16(random_si64(Int64(lo), Int64(hi)))
    return v


def reference(acc: I32x16, a: I16x32, b: I16x32) -> I32x16:
    """Scalar definition of the operation, the ground truth."""
    var r = acc
    for i in range(16):
        r[i] += (
            Int32(a[2 * i]) * Int32(b[2 * i])
            + Int32(a[2 * i + 1]) * Int32(b[2 * i + 1])
        )
    return r


def check() -> Bool:
    """Compares all three routes with the scalar reference.

    The ranges are the ones the real kernel will see: int16 activations
    anywhere in [-32767, 32767] and int4 weights in [-8, 7].
    """
    var ok = True
    for _ in range(1000):
        var acc = bitcast[DType.int32, 16](random_i16(-1000, 1000))
        var a = random_i16(-32767, 32767)
        var b = random_i16(-8, 7)
        var want = reference(acc, a, b)
        if dpwssd(acc, a, b) != want:
            ok = False
        if pmaddwd(acc, a, b) != want:
            ok = False
        if portable(acc, a, b) != want:
            ok = False
    return ok


# ===----------------------------------------------------------------------=== #
# Throughput
# ===----------------------------------------------------------------------=== #

# Each loop keeps N independent accumulators so the CPU can overlap
# instructions instead of waiting on one dependency chain.
#
# Benchmark pitfall (the first version of this file fell into it): if the
# inputs never change inside the loop, the compiler is free to compute
# `a * b` once, outside the loop, leaving only additions to time. pmaddwd and
# the portable version then looked 4x faster than f32 FMA, which the hardware
# cannot do. So `a` now comes from a small buffer that cycles through
# different vectors; it stays in L1 cache, so we still measure arithmetic
# rather than memory.
comptime N = 8
comptime ITERS = 20_000_000
comptime NBUF = 64  # vectors in the input buffer (4 KB)


def bench[
    Op: def(I32x16, I16x32, I16x32) -> I32x16
](name: StaticString, op: Op) -> Float64:
    """Runs `op` ITERS * N times; returns billions of multiply-adds per second.

    `Op` is a type parameter constrained by a function signature, so any
    function with that signature can be passed in as `op`. The compiler
    specializes `bench` for each function, and `op` is inlined into the loop.
    """
    var buf = unsafe_alloc[Int16](NBUF * 32)
    for i in range(NBUF):
        buf.unsafe_store(i * 32, random_i16(-100, 100))
    var b = Array[I16x32, length=N](fill=I16x32(0))
    comptime for k in range(N):
        b[k] = random_i16(-8, 7)
    var acc = Array[I32x16, length=N](fill=I32x16(0))
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        var a = buf.unsafe_load[width=32]((it % NBUF) * 32)
        comptime for k in range(N):
            acc[k] = op(acc[k], a, b[k])
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    # Print a result that depends on every accumulator, so the compiler can't
    # delete the loop as dead code.
    var total = I32x16(0)
    comptime for k in range(N):
        total += acc[k]
    buf.unsafe_free()
    var gmacs = Float64(ITERS * N * 32) / secs / 1e9
    print("  ", name, ":", gmacs, "GMAC/s  (checksum", total.reduce_add(), ")")
    return gmacs


def bench_f32_fma() -> Float64:
    """The baseline: float32 FMA, 16 multiply-adds per instruction."""
    var buf = unsafe_alloc[Float32](NBUF * 16)
    for i in range(NBUF * 16):
        buf[unsafe_offset=i] = Float32(i % 7) * 0.001
    var b = Array[F32x16, length=N](fill=F32x16(0))
    comptime for k in range(N):
        b[k] = F32x16(Float32(k) * 0.01)
    var acc = Array[F32x16, length=N](fill=F32x16(0))
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        var a = buf.unsafe_load[width=16]((it % NBUF) * 16)
        comptime for k in range(N):
            acc[k] = a.fma(b[k], acc[k])
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var total = F32x16(0)
    comptime for k in range(N):
        total += acc[k]
    buf.unsafe_free()
    var gmacs = Float64(ITERS * N * 16) / secs / 1e9
    print("   f32 fma :", gmacs, "GMAC/s  (checksum", total.reduce_add(), ")")
    return gmacs


def main():
    # CompilationTarget answers at compile time, for the CPU being compiled
    # for (by default, this machine).
    print("avx512f:", CompilationTarget.has_avx512f())
    print("vnni:   ", CompilationTarget.has_vnni())

    print("correctness vs scalar reference:", "PASS" if check() else "FAIL")

    print("single-thread throughput (inputs from L1 cache):")
    var f32 = bench_f32_fma()
    var vnni = bench("vpdpwssd", dpwssd)
    var madd = bench("pmaddwd ", pmaddwd)
    var port = bench("portable", portable)
    print("speedup over f32 fma: vpdpwssd", vnni / f32, "| pmaddwd", madd / f32,
          "| portable", port / f32)
