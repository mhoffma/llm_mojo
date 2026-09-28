# Typed tensors and W4A16 quantization for GPT-2 in Mojo

A learning project: GPT-2 124M inference on CPU, written from scratch in Mojo,
extended so the weights can be stored in different formats (float32, float16,
bfloat16, int8/int4 quantized) and eventually computed with integer
arithmetic (int4 weights × int16 activations → int32, using AVX-512 VNNI).

This file is the shared plan. Read it first, update the status and results
as milestones land, and add findings to the notes sections.

## Contents

1. [Why](#why)
2. [Status](#status)
3. [Getting started](#getting-started)
4. [Project layout](#project-layout)
5. [Verifying changes](#verifying-changes)
6. [Design](#design)
7. [Milestones](#milestones)
8. [Research findings](#research-findings)
9. [Mojo 1.1 notes](#mojo-11-notes)
10. [Conventions](#conventions)
11. [Results](#results)
12. [Open questions](#open-questions)

## Why

Generating text one token at a time reads every weight once per token, so
decoding is limited by **memory bandwidth**, not arithmetic. GPT-2 124M in
float32 is ~500 MB, which this laptop streams ~60 times per second, so ~60
tokens/s. Smaller weights mean proportionally fewer bytes and faster decoding:

| Weights | Size | Expected decode speedup |
|---|---|---|
| float32 | ~500 MB | 1× (baseline) |
| float16 / bfloat16 | ~250 MB | ~1.85× (measured on a streaming probe) |
| int8 | ~130 MB | ~3–4× |
| int4 | ~70 MB | higher, until compute becomes the limit |

Processing the prompt (prefill) is compute-bound instead, so there the win
comes from faster arithmetic: int16 VNNI does ~2.6× the multiply-adds of
float32 FMA on this CPU. The question the project answers is **how much
accuracy each format costs for that speed**.

## Status

| Milestone | State |
|---|---|
| Probes: VNNI, float16/bfloat16 | ✅ done (`research/`) |
| M1. Generic weight formats, float32 bit-identical to baseline | ✅ done |
| M2. Accuracy harness (perplexity, logit comparison) | ⏭ next |
| M3. float16 / bfloat16, then int4 (and int8) weights with float32 compute | ☐ |
| M4. int16 activations + integer VNNI kernel (W4A16) | ☐ |
| M5. Tuning and final results table | ☐ |
| M6. Stretch: pre-quantized weight files, int8 activations | ☐ |

## Getting started

### Requirements

- Linux on x86-64. Developed on an Intel i7-1160G7 (Tiger Lake, 4 cores /
  8 threads, AVX-512 with VNNI, F16C). Everything runs on any AVX2 CPU, but
  the VNNI kernel (M4) needs AVX-512 VNNI or AVX-VNNI; check with
  `grep -o 'avx512_vnni\|avx_vnni' /proc/cpuinfo | sort -u`.
- [uv](https://docs.astral.sh/uv/) (Python package manager).
- ~2 GB disk: ~1.1 GB for the environment, ~550 MB for the weights.
- Internet access for the first setup (PyPI and Hugging Face).

### Setup

```sh
cd ~/fun
uv sync            # creates .venv with mojo==1.1.0 and max==26.6.0 (pyproject.toml)
uv run mojo --version   # Mojo 1.1.0

# GPT-2 124M weights and tokenizer files from Hugging Face (~550 MB)
mkdir -p gpt2
for f in model.safetensors vocab.json merges.txt; do
  curl -L -o gpt2/$f https://huggingface.co/openai-community/gpt2/resolve/main/$f
done
```

Why the `max` package: in Mojo 1.1, `parallelize` moved out of the standard
library into `max.algorithm`. The `mojo` and `max` versions must match;
`max==26.6.0` is the one that pairs with `mojo==1.1.0`.

### Build and run

```sh
uv run mojo build -o gpt2_bin gpt2.mojo      # the float32 baseline
uv run mojo build -o gpt2t_bin gpt2t.mojo    # the generic-format version

export MODULAR_THREAD_BUSY_WAIT_US=0         # see note below
./gpt2t_bin --dtype f32 -n 100 "In a shocking finding, scientists discovered"
```

Options (both programs; `--dtype` only in `gpt2t`):

| Flag | Meaning | Default |
|---|---|---|
| `--dtype FMT` | weight format: `f32` (more per milestone) | `f32` |
| `-m DIR` | directory with the Hugging Face files | `gpt2` |
| `-n N` | tokens to generate | 64 |
| `-t TEMP` | sampling temperature; `0` = greedy | 0.8 |
| `-k K` | top-k sampling; `0` = all tokens | 40 |
| `-s SEED` | random seed | 1337 |
| `-v` | print prompt token ids and the top-5 next-token logits | off |

Timing and model size are printed to stderr after the generated text.

`uv run mojo run gpt2t.mojo ...` also works, but compiles every time (slow).

**`MODULAR_THREAD_BUSY_WAIT_US=0`**: by default the Mojo runtime's idle
worker threads spin between parallel regions. Decoding runs ~60 short parallel
regions per token, and on this 15 W, 4-core laptop the spinning threads take
cycles and power from the working ones: setting this to 0 took decode from
~38 to ~60 tok/s. The runtime reads it at startup, so it must be set in the
environment, not from inside the program. The best value may differ on other
machines.

### Benchmarking conditions

Numbers vary a lot with power state: on battery, long-context decode drops to
about half. Benchmark on AC power, with `MODULAR_THREAD_BUSY_WAIT_US=0`, and
interleave runs of the things you compare (A, B, A, B) rather than running
them back to back in blocks.

### Editor (optional)

`~/.emacs.d/mojo.el` gives Emacs a `mojo-mode` (python-mode plus Mojo
keywords) and connects `mojo-lsp-server` through lsp-mode. It finds the
server in the project's `.venv/bin`. Format-on-save is off by default. VS Code
users can install Modular's Mojo extension.

## Project layout

| Path | What |
|---|---|
| `gpt2.mojo` | **Frozen float32 baseline.** Single file, known correct. Don't change it; it's the reference the generic version is tested against. |
| `gpt2t.mojo` | The generic version: `Model[W]`, sampling, CLI, and `main`, which picks the format from `--dtype` |
| `tensor.mojo` | The `WeightMatrix` trait and the formats (`DenseMatrix[dtype]` so far) |
| `kernels.mojo` | matmul (prefill), GEMV (decode), LayerNorm, attention, output head; generic over `W: WeightMatrix` |
| `tokenizer.mojo` | GPT-2 byte-level BPE tokenizer (reads `vocab.json`, `merges.txt`) |
| `tests/` | Correctness checks (see below) |
| `research/` | Small commented probe programs, each answering one question; see `research/README.md` |
| `gpt2/` | Downloaded weights (not source) |
| `pyproject.toml`, `uv.lock` | The pinned environment |
| `aa.mojo` | Unrelated hello-world |

## Verifying changes

| Check | Command | Passes when |
|---|---|---|
| Generic float32 is bit-identical to the baseline | `tests/same_as_baseline.sh` | all lines `SAME` |
| Logits match an independent NumPy GPT-2 | `uv run python tests/reference.py logits 15496,11,616,1438,318` vs `./gpt2t_bin -v -n 0 "Hello, my name is"` | top-5 agree to ~1e-5 |
| Greedy decoding (KV cache path) matches NumPy | `uv run --with tiktoken python tests/reference.py greedy <ids> 30` vs `./gpt2t_bin -t 0 -n 30 "<prompt>"` | identical text |
| Tokenizer matches tiktoken | `uv run --with tiktoken python tests/tokenizer_vs_tiktoken.py` | `mismatches: 0` |

Build both binaries before running these. Run `tests/same_as_baseline.sh`
after any change to `gpt2t.mojo`, `kernels.mojo`, or `tensor.mojo`: float32
must stay bit-identical. Lossy formats are checked with the M2 accuracy
harness instead.

## Design

### Weight formats are a trait

Mojo has no class inheritance. A **trait** is an interface, and code generic
over it is compiled separately for each concrete type with its methods
inlined, so the abstraction costs nothing in the inner loops (M1 measured it:
the generic float32 build is as fast as the baseline).

```mojo
trait WeightMatrix(Deinitable, ImplicitlyCopyable):
    comptime NAME: StaticString                      # the --dtype name
    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int) -> Self   # convert at load time
    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]
    def nbytes(self) -> Int
    def free(self)
```

- Values are **handles** (a pointer plus shape). Copying copies the handle;
  `Model` owns them and calls `free` once in its destructor.
- `load` returns float32. Activations and arithmetic stay float32 until M4.
- `main` turns the runtime `--dtype` string into a compile-time type once:
  `run[DenseMatrix[DType.float32]](args)`. Everything under `run` is
  specialized for that format.

### What gets converted

The per-layer matrices (qkv `[768, 2304]`, attention projection `[768, 768]`,
MLP up `[768, 3072]`, MLP down `[3072, 768]`) and `wte` `[50257, 768]`, the
token embedding that doubles as the output head. These are ~99% of the bytes
read per token (`wte` alone is ~30%). LayerNorm parameters, biases, and the
position embedding `wpe` stay float32 in one small buffer.

### Layouts and kernels

- Layer matrices keep GPT-2's Conv1D layout `[IN, OUT]`: each input row
  contributes a contiguous slice of outputs, which suits SIMD along OUT.
- **Prefill** (`matmul`, T > 1 tokens): threads split the output columns
  into 32-wide blocks; each block processes 8 token rows at a time, keeping an
  8×32 tile of sums in registers so each weight load is reused 8 times.
- **Decode** (`gemv`, T = 1): threads split the *input rows*, so each thread
  streams whole contiguous rows (best for memory bandwidth), then the partial
  sums are added.
- **Output head** (`lm_head`): dot products against rows of `wte` `[V, C]`.
- Attention and the KV cache are float32. For short-context decode, attention
  runs on one thread, because waking the worker threads costs more.

## Milestones

### M1. Generic weight formats ✅

Trait + `DenseMatrix[float32]` + generic kernels + `Model[W]`. Acceptance:
`tests/same_as_baseline.sh` all `SAME`, and no speed loss. Both met.

### M2. Accuracy harness ⏭

Lossy formats need a number to judge them by, before they exist, so a
quantization bug can be told apart from quantization error.

- `--ppl FILE`: perplexity over a text file. Needs logits at *every*
  position, not just the last: add a forward mode that runs `lm_head` on all
  rows, and process the text in windows of up to 1024 tokens (e.g. stride
  512, scoring only the second half of each window after the first).
- `--compare FMT`: run the same text through float32 and FMT and report the
  top-1 agreement rate and the max / mean absolute logit difference.
- Choose and commit a standard evaluation text (a few thousand tokens of
  public-domain prose, e.g. from Project Gutenberg) under `tests/data/`.
- Acceptance: float32 perplexity matches the NumPy reference (add a `ppl`
  mode to `tests/reference.py`) to ~4 significant digits.

### M3. 16-bit, then int4 / int8 weights (float32 compute)

1. float16 and bfloat16: add `DenseMatrix[DType.float16]` and
   `DenseMatrix[DType.bfloat16]` branches in `main`. Measure.
2. Affine int4 and int8 in a new `QuantMatrix[bits, group, symmetric]`:
   - `x ≈ scale · (q − zero)`, with one scale (and zero point, if
     asymmetric) per group of `group` input rows for each output column.
     `group = IN` is **per-channel**.
   - int4: two values per byte.
   - `load` dequantizes to float32. Start with this generic path (correct
     first), then specialize `gemv` / `matmul` for it if it's slow: within a
     group, sum `x · q`, then apply the scale and zero point once per group.
   - Quantization happens in `from_f32` at load time. Start with plain
     round-to-nearest; GPTQ/AWQ-style methods are out of scope unless
     round-to-nearest is too inaccurate.
3. Sweep: per-channel vs group 128 / 64 / 32; symmetric vs asymmetric; `wte`
   in int4 vs int8 vs float16 (output heads are usually the most sensitive).
   This isolates the error from weight quantization alone.

Expectation to test: per-channel is fine for int8, but at int4 groups of
32–128 are usually needed for a model this small.

### M4. int16 activations + integer kernel (W4A16)

- Quantize only the matmul *inputs* (LayerNorm outputs, attention output,
  GELU output) to int16, with a scale per token row computed at runtime from
  its max absolute value. The residual stream, attention and KV cache stay
  float32.
- Kernel: unpack int4 weights to int16 in registers and accumulate
  `int16 × int16 → int32` with VPDPWSSD (see `research/test_vnni.mojo` for
  the call). Per group: one int32 → float32 conversion and an FMA with the
  group's scale. The zero point costs `zero · Σx_q` per group, with `Σx_q`
  computed once per token row.
- Overflow is impossible: |product| ≤ 32767 · 8 ≈ 2.6e5, and the sum over the
  largest input dimension (3072) is ≤ 8e8 < 2³¹.
- Compare W4A16 against the M3 W4A32 result to measure the error the
  activations add (expected: almost none).

### M5. Tuning and results

Kernel tuning for the quantized paths (tile sizes, prefetching, thread
split), then fill in the [Results](#results) table.

### M6. Stretch

- Save pre-quantized weights to a file so loading skips quantization.
- int8 activations with VPDPBUSD (u8 × s8), which needs care with GPT-2's
  outlier activation channels.
- Quantized KV cache for long contexts.

## Research findings

The probes in `research/` (run: `uv run mojo run research/<file>.mojo`):

- **`test_vnni.mojo`**: Mojo 1.1 can call VNNI:
  `llvm_intrinsic["llvm.x86.avx512.vpdpwssd.512", SIMD[DType.int32, 16], has_side_effect=False](acc, a, b)`
  where `a`, `b` are `SIMD[DType.int16, 32]` (LLVM now declares them as
  `<32 x i16>`, not the older `<16 x i32>`). Exact against a scalar
  reference. One thread: f32 FMA ~35 GMAC/s, VPDPWSSD ~92 GMAC/s (~2.6×).
  LLVM fuses VPMADDWD + add into VPDPWSSD itself; plain portable SIMD code is
  *not* recognized and runs ~20 GMAC/s, so call the intrinsic directly.
- **`test_half.mojo`**: this CPU has F16C (conversion only), no AVX512_FP16,
  no AVX512_BF16, no AMX. So 16-bit floats are storage formats, widened to
  float32 in registers (float16: VCVTPH2PS; bfloat16: a 16-bit shift).
  Widening costs ~11% when compute-bound; streaming 16-bit weights from memory
  gives ~1.85× the weights/s of float32. For GPT-2-sized weights float16 is
  ~8× more precise than bfloat16.

Hardware of the development machine (i7-1160G7): 4 cores / 8 threads, one
512-bit FMA unit per core, 5 MB L2, 12 MB L3, 16 GB RAM, measured ~44–55 GB/s
read bandwidth from memory with all threads.

## Mojo 1.1 notes

Mojo changes quickly and most online examples are for older versions. The
exact stdlib source for this version is the tag `mojo/v1.1.0` of
[github.com/modular/modular](https://github.com/modular/modular), under
`Mojo/stdlib/std` (a sparse checkout is quick):

```sh
git clone --depth 1 --branch mojo/v1.1.0 --filter=blob:none --sparse \
  https://github.com/modular/modular mojo-src
cd mojo-src && git sparse-checkout set Mojo/stdlib/std
```

Things that differ from older Mojo, all hit during this project:

| Topic | Mojo 1.1 |
|---|---|
| Imports | stdlib modules need the `std.` prefix: `from std.math import exp` |
| Parallel loops | `from max.algorithm import parallelize`; thread count: `from std.runtime import parallelism_level` |
| Allocation | `from std.memory.alloc import unsafe_alloc`; `p.unsafe_free()` |
| Pointers | `Pointer[T, MutUntrackedOrigin]` (UnsafePointer is a deprecated alias); index with `p[unsafe_offset=i]`; SIMD with `p.unsafe_load[width=W](i)` / `p.unsafe_store(i, v)`; offset with `p.unsafe_offset(n)`, not `p + n` |
| Errors | `def` does not raise unless marked `def f() raises:` |
| Closures | must declare captures: `def body(i: Int) {imm}:` (or `{mut}`) |
| Strings | `len(s)` is an error; use `s.byte_length()` or `len(s.as_bytes())` |
| Fixed arrays | `Array[T, length=N](fill=x)` (was InlineArray) |
| Compile time | `comptime X = ...` (was alias), `comptime for`, `comptime if` (was `@parameter`) |
| Struct parameters | inside the struct, write `Self.W`, not `W` |
| Results | a function can't have both an `out` argument and a return type; use `mut` |
| Comptime arrays | bind an element before runtime use: `comptime n = SIZES[i]` |
| Traits for List elements | must include `Deinitable` |
| Functions as parameters | `def f[Op: def(A) -> B](op: Op)`, then call `f(my_func)` |
| Local modules | sibling `.mojo` files import directly: `from tensor import WeightMatrix` |
| Assembly | `uv run mojo build --emit asm -o out.s file.mojo` |

## Conventions

- `gpt2.mojo` is frozen. New work goes in `gpt2t.mojo`, `tensor.mojo`,
  `kernels.mojo`.
- float32 through the generic code must stay bit-identical to the baseline
  (`tests/same_as_baseline.sh`).
- Every new format gets numbers in the [Results](#results) table: size,
  prefill tok/s, decode tok/s, perplexity, top-1 agreement.
- When a question comes up that needs an experiment, write a small commented
  probe in `research/test_<topic>.mojo` whose docstring states the question,
  how to run it, the results, and the conclusion, and add it to
  `research/README.md`. They're meant for learning Mojo as well as for
  answers.
- Keep diffs focused; don't reformat whole files you didn't otherwise change,
  and don't add automatic formatting (format-on-save or similar).
- Record benchmark conditions (AC power, busy-wait setting) with the numbers.

## Results

On the i7-1160G7, AC power, `MODULAR_THREAD_BUSY_WAIT_US=0`. Prefill: 476-token
prompt. Decode: 200 tokens after a short prompt / after the 476-token prompt.

| Format | Weights | Prefill tok/s | Decode tok/s (short / long ctx) | Perplexity | Top-1 vs f32 |
|---|---|---|---|---|---|
| f32 (baseline `gpt2.mojo`) | 474 MB | ~620 | ~62 / ~57 | M2 | 100% |
| f32 (`gpt2t`) | 474 MB | same | same | M2 | 100% (bit-identical) |
| f16 | | | | | |
| bf16 | | | | | |
| int8, per-channel | | | | | |
| int4, per-channel | | | | | |
| int4, group 128 | | | | | |
| int4, group 64 | | | | | |
| int4, group 32 | | | | | |
| W4A16 (best int4 config) | | | | | |

## Open questions

- Is per-channel int4 accurate enough for GPT-2 124M, or are groups needed?
  (M3 sweep)
- Can `wte` (embedding + output head) go to int4, or does it need int8 or
  float16?
- How close to the memory bandwidth limit can the int4 decode kernel get?
  Unpacking int4 costs instructions, and at ~70 MB per token the arithmetic
  may become the limit instead.
- Would transposing the layer matrices to `[OUT, IN]` (dot-product layout,
  as llama.cpp does) be better for the quantized kernels than keeping
  `[IN, OUT]`?
