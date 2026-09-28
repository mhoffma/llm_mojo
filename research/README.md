# Research probes

Small, self-contained Mojo programs that each answer one question for the
typed-tensor / W4A16 GPT-2 work. Each file's docstring states the question,
how to run it, and the results measured on this machine.

Run any of them from `~/fun` with:

    uv run mojo run research/<file>.mojo

| File | Question | Answer |
|---|---|---|
| `test_vnni.mojo` | Can Mojo 1.1 use AVX-512 VNNI (int16 × int16 → int32) and how fast is it? | Yes, via `llvm_intrinsic["llvm.x86.avx512.vpdpwssd.512"]` with `<32 x i16>` operands. ~2.6× the multiply-adds of f32 FMA. |
| `test_half.mojo` | Does this CPU support float16/bfloat16, and what do 16-bit weights cost and save? | Storage only (F16C conversion, no 16-bit math). Widening costs ~11% when compute-bound; streaming weights from DRAM is ~1.85× faster. float16 is 8× more precise than bfloat16 for GPT-2-sized weights. |

Mojo concepts shown along the way:

- `test_vnni.mojo`: SIMD types, `llvm_intrinsic`, `comptime for` unrolling,
  passing functions as parameters, `CompilationTarget` feature checks,
  and a benchmarking pitfall (loop-invariant hoisting).
- `test_half.mojo`: float16/bfloat16 dtypes, `.cast` vs `bitcast`, generic
  functions over `DType`, `comptime if`, and a multi-threaded streaming
  benchmark with `parallelize`.
