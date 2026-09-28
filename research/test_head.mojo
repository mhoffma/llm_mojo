"""Probe: how fast can the output head go?

GPT-2's output head is a GEMV with 50257 x 768 = 38.6M weights, the biggest
single operation of a decode step (16-35% of it). Its weights are far more
than the 12 MB L3 holds, so each token streams them all from DRAM.

Part 1, our int8 head (int formats). ~38.6 MB per token, so it should be
limited by memory bandwidth, yet the model took 1.07 ms (~36 GB/s). This
compares, at 4 threads (one per core, the model's team) and 8 (both
hyperthreads of each core):

  read     a pure read of the same bytes: the bandwidth ceiling
  row      one row at a time, codes widened to int16 minus the zero point,
           VPDPWSSD with int16 activations, loop fully unrolled
  row4     four rows at a time, sharing each activation load
  model    QuantMatrix[8, 0, False, True].dot_row_i16, the model's kernel
  pf N     `row` plus a software prefetch of the row N rows ahead

The rows have the model's layout: [V, C] uint8 codes, one zero point and
scale per row (per-channel), int16 activations.

Part 2, GGUF heads. Every GPT-2 GGUF file stores the head as Q6_K (Q4_0,
Q4_K_M) or Q8_0 (Q8_0). gguf.mojo's float kernels convert each weight to
float32 and multiply-add it: limited by arithmetic, not memory. On the real
Q6_K head of Q4_K_M:

  float     GGUFMatrix.dot_row (the float kernel)
  model     GGUFMatrix.dot_row_i16: int16 activations, int16 weights
            (q - 32) * scale, VPDPWSSD; the scale vector for each 32 weights
            is a shuffle of the superblock's 16 scales
  broadcast the same, scale vectors from two scalar loads + broadcasts
  vpermw    the same, but forcing one VPERMW per scale vector (the compiler
            rewrites `model`'s constant-mask shuffle into two scalar loads,
            two moves and an insert before the VPERMW; see the assembly:
            uv run mojo build -I . --emit asm -o /tmp/head.s research/test_head.mojo)
  read      a pure read of the same bytes

Run: MODULAR_THREAD_BUSY_WAIT_US=0 uv run mojo run -I . research/test_head.mojo
     (part 2 needs gpt2/gguf/gpt2.Q4_K_M.gguf, see PLAN.md > Setup)

Results on the i7-1160G7, 2026-09-28, ms per call (median of 9 rounds of
10 calls; each call includes a parallel region, ~23 us; this laptop's
numbers move by ~5-10% between runs):

  int8 head                 4 threads   8 threads
    read                    0.77-0.86   0.80-0.84   45-50 GB/s
    row                     0.91-0.99   0.85-0.91
    row4                    0.98-1.03   0.93-1.01
    model, before           1.07-1.13   0.95-1.10
    model, now              0.96-1.00   0.95-1.00
    pf 8                    0.84-0.93   0.86-0.92

  Q6_K head (31.7 MB)
    float                   1.88-2.01   1.91-2.35   ~16 GB/s
    broadcast               1.34-1.42   1.41-1.59
    model                   1.16-1.23   1.12-1.26   ~27 GB/s
    vpermw                  1.16-1.24   1.11-1.33
    read                    0.57-0.61   0.54-0.69   ~52 GB/s

  int16 vs float Q6_K logits: mean |diff| ~1e-3 (logits up to 30), same
  argmax; broadcast, vpermw and model give identical results.

Conclusions:
- The int8 head is memory-bound, within ~15% of a plain read. The model's
  kernel was ~15% slower than the unrolled `row`: its loop over the row had
  a runtime trip count. Unrolling it 4x and prefetching 8 rows ahead
  (tensor.mojo, per-channel int8) brought it to `row`'s speed. More threads
  or more rows at a time don't help.
- The Q6_K head in integers is ~1.7x faster than in float. It is still
  limited by arithmetic, not memory (~2x the read time): ~8 vector
  instructions per 32 weights (unpacking 6-bit codes, widening, the scale
  multiply, VPDPWSSD) at the ~1.5 GHz this 15 W chip runs AVX-512 on all
  cores. The next step would be int8 activations with VPDPBUSD (64 weights
  per instruction, no widening), as llama.cpp does, at a cost in precision.
- Building each scale vector with a shuffle instead of broadcasts: ~12%.
  Forcing a true VPERMW beyond that: no measurable gain, so gguf.mojo uses
  the plain shuffle.
- In the model: int8 head 1.10 -> 1.00 ms; GGUF heads Q6_K 2.1-2.8 ->
  1.3-1.5 ms, Q8_0 2.6 -> 1.8 ms (see PLAN.md, M6).
"""

from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns
from std.random import random_ui64
from std.sys.info import num_physical_cores
from std.sys.intrinsics import prefetch, llvm_intrinsic
from max.algorithm import parallelize

from tensor import dot_pairs, I16x32, I32x16, F32x16, QuantMatrix, I16Ptr
from gguf import GGUFFile, GGUFMatrix, f16_at
from kernels import quantize_rows

comptime V = 50257
comptime C = 768
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I16P = Pointer[Int16, MutUntrackedOrigin]
comptime FP = Pointer[Float32, MutUntrackedOrigin]

comptime READ = 0
comptime ROW = 1
comptime ROW4 = 2
comptime PF = 3
comptime QM = 4

comptime Int8Head = QuantMatrix[8, 0, False, True]


@always_inline
def dot_row(w: U8Ptr, zp: U8Ptr, sc: FP, x: I16P, v: Int) -> Float32:
    """The model's per-channel int8 dot: sum_k (u[k] - zp) * x[k] * scale."""
    var p = w.unsafe_offset(v * C)
    var z = I16x32(Int16(Int(zp[unsafe_offset=v])))
    var acc = I32x16(0)
    comptime for k in range(0, C, 32):
        var u = p.unsafe_load[width=32](k).cast[DType.int16]() - z
        acc = dot_pairs(acc, x.unsafe_load[width=32](k), u)
    return Float32(acc.reduce_add()) * sc[unsafe_offset=v]


@always_inline
def dot_row4(w: U8Ptr, zp: U8Ptr, sc: FP, x: I16P, v: Int, out_: FP):
    """Rows v..v+3 at once: each activation vector is loaded once for four
    rows, and four independent accumulators keep VPDPWSSD busy."""
    var a0 = I32x16(0)
    var a1 = I32x16(0)
    var a2 = I32x16(0)
    var a3 = I32x16(0)
    var z0 = I16x32(Int16(Int(zp[unsafe_offset=v])))
    var z1 = I16x32(Int16(Int(zp[unsafe_offset=v + 1])))
    var z2 = I16x32(Int16(Int(zp[unsafe_offset=v + 2])))
    var z3 = I16x32(Int16(Int(zp[unsafe_offset=v + 3])))
    var p = w.unsafe_offset(v * C)
    comptime for k in range(0, C, 32):
        var xk = x.unsafe_load[width=32](k)
        a0 = dot_pairs(a0, xk, p.unsafe_load[width=32](k).cast[DType.int16]() - z0)
        a1 = dot_pairs(a1, xk, p.unsafe_load[width=32](C + k).cast[DType.int16]() - z1)
        a2 = dot_pairs(a2, xk, p.unsafe_load[width=32](2 * C + k).cast[DType.int16]() - z2)
        a3 = dot_pairs(a3, xk, p.unsafe_load[width=32](3 * C + k).cast[DType.int16]() - z3)
    out_[unsafe_offset=v] = Float32(a0.reduce_add()) * sc[unsafe_offset=v]
    out_[unsafe_offset=v + 1] = Float32(a1.reduce_add()) * sc[unsafe_offset=v + 1]
    out_[unsafe_offset=v + 2] = Float32(a2.reduce_add()) * sc[unsafe_offset=v + 2]
    out_[unsafe_offset=v + 3] = Float32(a3.reduce_add()) * sc[unsafe_offset=v + 3]


def run[KIND: Int, AHEAD: Int = 4](
    w: U8Ptr, zp: U8Ptr, sc: FP, x: I16P, out_: FP, nt: Int, qm: Int8Head
) -> Float64:
    """Mean time in ms of 10 calls of one head GEMV (or read) on nt
    threads."""
    var total = Float64(0)
    for _ in range(10):
        var t0 = perf_counter_ns()

        def part(t: Int) {imm}:
            var per = (V + nt - 1) // nt
            per = (per + 3) // 4 * 4  # multiples of 4 rows for row4
            var r0 = min(V, t * per)
            var r1 = min(V, (t + 1) * per)
            comptime if KIND == READ:
                var acc = SIMD[DType.uint64, 8](0)
                var p = w.unsafe_offset(r0 * C)
                for i in range(0, (r1 - r0) * C, 64):
                    # XOR the 64 bytes in as 8 uint64: one cheap instruction
                    # per cache line, so only the reads count.
                    acc ^= bitcast[DType.uint64, 8](p.unsafe_load[width=64](i))
                out_[unsafe_offset=r0] = Float32(acc.reduce_add())
            elif KIND == ROW:
                for v in range(r0, r1):
                    out_[unsafe_offset=v] = dot_row(w, zp, sc, x, v)
            elif KIND == ROW4:
                var v = r0
                while v + 4 <= r1:
                    dot_row4(w, zp, sc, x, v, out_)
                    v += 4
                while v < r1:
                    out_[unsafe_offset=v] = dot_row(w, zp, sc, x, v)
                    v += 1
            elif KIND == QM:
                for v in range(r0, r1):
                    out_[unsafe_offset=v] = qm.dot_row_i16(v, x, sc)
            else:
                for v in range(r0, r1):
                    # Each 768-byte row is 12 cache lines: prefetch the row
                    # AHEAD rows ahead, line by line.
                    var q = w.unsafe_offset((v + AHEAD) * C)
                    comptime for l in range(12):
                        prefetch(q.unsafe_offset(l * 64))
                    out_[unsafe_offset=v] = dot_row(w, zp, sc, x, v)

        parallelize(part, nt)
        total += Float64(perf_counter_ns() - t0) / 1e6
    return total / 10


# ===----------------------------------------------------------------------=== #
# Part 2: the GGUF Q6_K head (every GPT-2 GGUF file stores the head as Q6_K)
# ===----------------------------------------------------------------------=== #

comptime Q6_FLOAT = 0  # GGUFMatrix.dot_row: float32 dequantize + FMA
comptime Q6_INT = 1  # GGUFMatrix.dot_row_i16: int16 VPDPWSSD, scales by shuffle
comptime Q6_BCAST = 2  # the same, scales from two scalar loads + broadcasts
comptime Q6_PERMW = 4  # the same, forcing one VPERMW per scale vector
comptime Q6_READ = 3  # read the bytes only


@always_inline
def q6_bcast[n: Int, k: Int](b: U8Ptr) -> I16x32:
    """The first version of gguf.q6_k_i16: each scale vector built from two
    scalar loads, each broadcast to 16 lanes."""
    var L = b.unsafe_load[width=32](64 * n + (32 if k % 2 == 1 else 0))
    var H = b.unsafe_load[width=32](128 + 32 * n)
    comptime SHIFT = SIMD[DType.uint8, 32](2 * k)
    var q = (((L >> 4) if k >= 2 else (L & 0xF)) | (((H >> SHIFT) & 3) << 4))
    var s0 = Int16(Int(b[unsafe_offset = 192 + 8 * n + 2 * k].cast[DType.int8]()))
    var s1 = Int16(Int(b[unsafe_offset = 193 + 8 * n + 2 * k].cast[DType.int8]()))
    var sc = SIMD[DType.int16, 16](s0).join(SIMD[DType.int16, 16](s1))
    return (q.cast[DType.int16]() - 32) * sc


@always_inline
def q6_permw[n: Int, k: Int](b: U8Ptr, sv: I16x32, z: I16x32) -> I16x32:
    """As gguf.q6_k_i16 (a shuffle of the 16 loaded scales), but with the
    shuffle's mask hidden behind z, a zero the compiler can't see. With a
    constant mask, the compiler notices each shuffle needs just 2 of the 16
    loaded scales and rewrites it as two scalar loads, two moves and an
    insert before the VPERMW (see the assembly); this keeps one VPERMW."""
    var L = b.unsafe_load[width=32](64 * n + (32 if k % 2 == 1 else 0))
    var H = b.unsafe_load[width=32](128 + 32 * n)
    comptime SHIFT = SIMD[DType.uint8, 32](2 * k)
    var q = (((L >> 4) if k >= 2 else (L & 0xF)) | (((H >> SHIFT) & 3) << 4))
    comptime MASK = SIMD[DType.int16, 16](Int16(8 * n + 2 * k)).join(
        SIMD[DType.int16, 16](Int16(8 * n + 2 * k + 1))
    )
    var sc = llvm_intrinsic[
        "llvm.x86.avx512.permvar.hi.512", I16x32, has_side_effect=False
    ](sv, MASK + z)
    return (q.cast[DType.int16]() - 32) * sc


@always_inline
def dot_q6[PERMW: Bool](m: GGUFMatrix, row: Int, xq: I16Ptr) -> Float32:
    """GGUFMatrix.dot_row_i16 with q6_bcast or q6_permw."""
    var p = m.data.unsafe_offset(row * m.row_bytes)
    var z = I16x32(Int16(row >> 62))  # 0 for any real row
    var accf = F32x16(0)
    for sb in range(m.cols // 256):
        var b = p.unsafe_offset(sb * 210)
        var s16 = b.unsafe_load[width=16](192).cast[DType.int8]().cast[DType.int16]()
        var sv = s16.join(s16)
        var acc = I32x16(0)
        comptime for n in range(2):
            comptime for k in range(4):
                var w: I16x32
                comptime if PERMW:
                    w = q6_permw[n, k](b, sv, z)
                else:
                    w = q6_bcast[n, k](b)
                acc = dot_pairs(
                    acc, xq.unsafe_load[width=32](sb * 256 + 128 * n + 32 * k), w
                )
        accf = acc.cast[DType.float32]().fma(F32x16(f16_at(b, 208)), accf)
    return accf.reduce_add()


def run_q6[KIND: Int](
    m: GGUFMatrix, x: FP, xq: I16Ptr, s: Float32, out_: FP, nt: Int
) -> Float64:
    """Mean time in ms of 10 calls of the Q6_K head on nt threads."""
    var total = Float64(0)
    for _ in range(10):
        var t0 = perf_counter_ns()

        def part(t: Int) {imm}:
            var per = (m.rows + nt - 1) // nt
            var r0 = min(m.rows, t * per)
            var r1 = min(m.rows, (t + 1) * per)
            comptime if KIND == Q6_READ:
                var acc = SIMD[DType.uint64, 8](0)
                var p = m.data.unsafe_offset(r0 * m.row_bytes)
                var n = (r1 - r0) * m.row_bytes
                for i in range(0, n - 63, 64):
                    acc ^= bitcast[DType.uint64, 8](p.unsafe_load[width=64](i))
                out_[unsafe_offset=r0] = Float32(acc.reduce_add())
            else:
                for v in range(r0, r1):
                    comptime if KIND == Q6_FLOAT:
                        out_[unsafe_offset=v] = m.dot_row(v, x)
                    elif KIND == Q6_INT:
                        out_[unsafe_offset=v] = m.dot_row_i16(v, xq, x) * s
                    elif KIND == Q6_BCAST:
                        out_[unsafe_offset=v] = dot_q6[False](m, v, xq) * s
                    else:
                        out_[unsafe_offset=v] = dot_q6[True](m, v, xq) * s

        parallelize(part, nt)
        total += Float64(perf_counter_ns() - t0) / 1e6
    return total / 10


def q6_head() raises:
    var g = GGUFFile("gpt2/gguf/gpt2.Q4_K_M.gguf")
    var m = g.matrix("output.weight")
    var mb = Float64(m.rows * m.row_bytes) / 1e6
    print()
    print("Q6_K head (Q4_K_M output.weight),", m.rows, "x", m.cols, "=", mb, "MB")
    # An activation vector shaped like ln_f's output: mostly small values
    # and a few large outlier channels.
    var x = unsafe_alloc[Float32](m.cols)
    for i in range(m.cols):
        x[unsafe_offset=i] = Float32(Int(random_ui64(0, 2000)) - 1000) * 1e-3
    x[unsafe_offset=138] = 25
    x[unsafe_offset=447] = -40
    var xq = unsafe_alloc[Int16](m.cols)
    var sx = unsafe_alloc[Float32](1)
    quantize_rows(x, 1, m.cols, xq, sx)
    var s = sx[unsafe_offset=0]
    var op = unsafe_alloc[Float32](m.rows)
    var of = unsafe_alloc[Float32](m.rows)
    var oi = unsafe_alloc[Float32](m.rows)
    var os = unsafe_alloc[Float32](m.rows)

    comptime ROUNDS = 9
    var names: List[String] = ["float    ", "model    ", "broadcast", "read     ", "vpermw   "]
    var nc = num_physical_cores()
    for nt in [nc, 2 * nc]:
        var ms = List[List[Float64]]()
        for _ in range(5):
            ms.append(List[Float64]())
        for _ in range(ROUNDS):
            ms[0].append(run_q6[Q6_FLOAT](m, x, xq, s, of, nt))
            ms[1].append(run_q6[Q6_INT](m, x, xq, s, oi, nt))
            ms[2].append(run_q6[Q6_BCAST](m, x, xq, s, os, nt))
            ms[3].append(run_q6[Q6_READ](m, x, xq, s, os, nt))
            ms[4].append(run_q6[Q6_PERMW](m, x, xq, s, op, nt))
        for k in range(5):
            sort(ms[k])
            var t = ms[k][ROUNDS // 2]
            print("  ", names[k], nt, "threads:", t, "ms ", mb / t, "GB/s")
    # Accuracy of the int16 path against the float kernel, and the
    # broadcast and vpermw variants against it (should be identical).
    var err = Float64(0)
    var mx = Float64(0)
    var top_f = 0
    var top_i = 0
    var same = 0
    _ = run_q6[Q6_BCAST](m, x, xq, s, os, nc)
    for v in range(m.rows):
        var d = abs(Float64(oi[unsafe_offset=v] - of[unsafe_offset=v]))
        err += d
        mx = max(mx, abs(Float64(of[unsafe_offset=v])))
        if of[unsafe_offset=v] > of[unsafe_offset=top_f]:
            top_f = v
        if oi[unsafe_offset=v] > oi[unsafe_offset=top_i]:
            top_i = v
        if os[unsafe_offset=v] == oi[unsafe_offset=v] and op[unsafe_offset=v] == oi[unsafe_offset=v]:
            same += 1
    print("   model (int16) vs float: mean |diff|", err / Float64(m.rows), "(max |logit|", mx,
          ") argmax", top_i, "vs", top_f)
    print("   broadcast == vpermw == model in", same, "of", m.rows, "rows")
    # m points into g's buffer: keep g alive until here (Mojo destroys a
    # value right after its last use, here g.matrix).
    _ = g^

def main() raises:
    # One extra row of 4 ahead so prefetches past the end stay in bounds.
    var w = unsafe_alloc[UInt8]((V + 8) * C)
    for i in range((V + 8) * C):
        w[unsafe_offset=i] = UInt8(Int(random_ui64(0, 255)))
    var zp = unsafe_alloc[UInt8](V + 8)
    var sc = unsafe_alloc[Float32](V + 8)
    for v in range(V + 8):
        zp[unsafe_offset=v] = UInt8(Int(random_ui64(100, 155)))
        sc[unsafe_offset=v] = 0.001
    var x = unsafe_alloc[Int16](C)
    for k in range(C):
        x[unsafe_offset=k] = Int16(Int(random_ui64(0, 2000))) - 1000
    var o1 = unsafe_alloc[Float32](V + 8)
    var o2 = unsafe_alloc[Float32](V + 8)

    # The model's int8 head format, quantized from random float weights.
    var wf = unsafe_alloc[Float32](V * C)
    for i in range(V * C):
        wf[unsafe_offset=i] = Float32(Int(random_ui64(0, 2000)) - 1000) * 1e-4
    var qm = Int8Head.from_f32(wf, V, C, False)
    wf.unsafe_free()

    # The variants alternate over ROUNDS rounds of 10 calls each; each
    # round's mean is one sample and the median sample is reported, which
    # evens out this laptop's turbo and thermal swings.
    comptime ROUNDS = 9
    comptime NV = 7
    var names: List[String] = ["read ", "row  ", "row4 ", "model", "pf 2 ", "pf 4 ", "pf 8 "]
    var nc = num_physical_cores()
    print("int8 head GEMV,", V, "x", C, "=", Float64(V * C) / 1e6, "MB per call")
    print("median over", ROUNDS, "rounds of the mean of 10 calls")
    for nt in [nc, 2 * nc]:
        var ms = List[List[Float64]]()
        for _ in range(NV):
            ms.append(List[Float64]())
        for _ in range(ROUNDS):
            ms[0].append(run[READ](w, zp, sc, x, o1, nt, qm))
            ms[1].append(run[ROW](w, zp, sc, x, o1, nt, qm))
            ms[2].append(run[ROW4](w, zp, sc, x, o2, nt, qm))
            ms[3].append(run[QM](w, zp, sc, x, o2, nt, qm))
            ms[4].append(run[PF, 2](w, zp, sc, x, o2, nt, qm))
            ms[5].append(run[PF, 4](w, zp, sc, x, o2, nt, qm))
            ms[6].append(run[PF, 8](w, zp, sc, x, o2, nt, qm))
        for k in range(NV):
            sort(ms[k])
            var t = ms[k][ROUNDS // 2]
            print("  ", names[k], nt, "threads:", t, "ms ",
                  Float64(V * C) / t / 1e6, "GB/s")
        # Same outputs? row vs row4 and pf (o1 holds row's, o2 pf 8's).
        var bad = 0
        for v in range(V):
            if o1[unsafe_offset=v] != o2[unsafe_offset=v]:
                bad += 1
        print("     pf differs from row in", bad, "rows")
    q6_head()
