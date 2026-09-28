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
| `test_dequant.mojo` | How fast does `gguf.mojo` dequantize each GGUF block format, and does fusing the dot product help? | 8–19 G weights/s per thread dequantizing to a buffer; reading it back for the dot halves that; fusing the dot into the dequantizer is 1.3–2.5× faster than dequantize-then-dot. |
| `test_softmax.mojo` | Can softmax(x + mask) run in integers (e^x as 2^x: shift plus a polynomial for the fraction), and is it accurate and fast? | Yes, accurately: degree-3 polynomial, Q15 output, rounding: max error ~1 Q15 step, KL 5.7e-4. SIMD matches scalar exactly. But on this CPU it's slower (0.88 ns/element) than a SIMD float softmax (0.40); both beat the scalar-exp float softmax attention uses now (5.8). |

Mojo concepts shown along the way:

- `test_vnni.mojo`: SIMD types, `llvm_intrinsic`, `comptime for` unrolling,
  passing functions as parameters, `CompilationTarget` feature checks,
  and a benchmarking pitfall (loop-invariant hoisting).
- `test_half.mojo`: float16/bfloat16 dtypes, `.cast` vs `bitcast`, generic
  functions over `DType`, `comptime if`, and a multi-threaded streaming
  benchmark with `parallelize`.
- `test_dequant.mojo`: importing project modules (`-I .`), timing kernels
  on real GGUF tensors, returning tuples.
- `test_softmax.mojo`: fixed-point arithmetic (Q formats, multipliers with
  extra fraction bits), a masked-softmax kernel in scalar and SIMD form,
  `select` for branch-free masking, int64 SIMD.
