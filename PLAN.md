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
| M2. Accuracy harness (perplexity, logit comparison) | ✅ done |
| M3. float16 / bfloat16, int8/int4 weights, GGUF weights; float32 compute | ✅ done (f16/bf16, GGUF, own int8/int4 with int8 head, fast kernels for all) |
| M4. int16 activations + integer VNNI kernel (W4A16) | ✅ done: int4-g32-a16 is the fastest decoder (~140 tok/s) at +1.2% perplexity, prefill ~800 tok/s |
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

# Pre-quantized llama.cpp GGUF files of the same model (~100-180 MB each)
mkdir -p gpt2/gguf
curl -L -o gpt2/gguf/gpt2.Q4_K_M.gguf https://huggingface.co/mradermacher/gpt2-GGUF/resolve/main/gpt2.Q4_K_M.gguf
curl -L -o gpt2/gguf/gpt2.i1-Q4_K_M.gguf https://huggingface.co/mradermacher/gpt2-i1-GGUF/resolve/main/gpt2.i1-Q4_K_M.gguf
for q in Q4_0 Q4_1 Q8_0; do
  curl -L -o gpt2/gguf/gpt2.$q.gguf https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.$q.gguf
done
```

(If you script downloads in zsh, note it does not word-split unquoted
variables; `set -- $f` style loops need bash.)

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
| `--dtype FMT` | weight format: `f32`, `f16`, `bf16`, `int8-ch`, `int4-ch`, `int4-g128`, `int4-g64`, `int4-g32`, each int format also with `-sym` | `f32` |
| `--head FMT` | with an int `--dtype`: format of wte (embedding + output head), `int8` or `same` | `int8` |
| `-m DIR` | directory with the Hugging Face files | `gpt2` |
| `-n N` | tokens to generate | 64 |
| `-t TEMP` | sampling temperature; `0` = greedy | 0.8 |
| `-k K` | top-k sampling; `0` = all tokens | 40 |
| `-s SEED` | random seed | 1337 |
| `-v` | print prompt token ids and the top-5 next-token logits | off |
| `--gguf FILE` | use a llama.cpp GGUF file's weights as stored (overrides `--dtype`; `gpt2t` only) | |
| `--ppl FILE` | instead of generating, measure perplexity on a text file (`gpt2t` only) | |
| `--compare` | with `--ppl`: also run float32 and compare predictions (`gpt2t` only) | off |

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
| `README.md` | Project overview: features, results, setup, tensor structure, kernels |
| `gpt2t.mojo` | The generic version: `Model[W, E]`, loaders, sampling, evaluation, CLI, and `main`, which picks the formats from `--dtype` / `--head` / `--gguf` |
| `tensor.mojo` | The `WeightMatrix` trait and our formats: `DenseMatrix[dtype]`, `QuantMatrix[bits, group, symmetric, a16]` |
| `gguf.mojo` | GGUF file reader and `GGUFMatrix`: dequantizes Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, F32 |
| `kernels.mojo` | matmul (prefill), GEMV (decode), `matmul_rows` for `[OUT, IN]` weights, LayerNorm, attention, output head; generic over `W: WeightMatrix` |
| `tokenizer.mojo` | GPT-2 byte-level BPE tokenizer (reads `vocab.json`, `merges.txt`) |
| `tests/` | Correctness checks and the accuracy harness's reference (see below); `tests/data/` holds the evaluation text |
| `research/` | Small commented probe programs, each answering one question; see `research/README.md` |
| `gpt2/` | Downloaded weights (not source) |
| `pyproject.toml`, `uv.lock` | The pinned environment |
| `aa.mojo` | Unrelated hello-world |

## Verifying changes

| Check | Command | Passes when |
|---|---|---|
| Speed (not a pass/fail check) | `tests/bench.sh 3 f32 f16 gpt2/gguf/gpt2.Q4_K_M.gguf` | compare medians within one run |
| Generic float32 is bit-identical to the baseline | `tests/same_as_baseline.sh` | all lines `SAME` |
| Logits match an independent NumPy GPT-2 | `uv run python tests/reference.py logits 15496,11,616,1438,318` vs `./gpt2t_bin -v -n 0 "Hello, my name is"` | top-5 agree to ~1e-5 |
| Greedy decoding (KV cache path) matches NumPy | `uv run --with tiktoken python tests/reference.py greedy <ids> 30` vs `./gpt2t_bin -t 0 -n 30 "<prompt>"` | identical text |
| Tokenizer matches tiktoken | `uv run --with tiktoken python tests/tokenizer_vs_tiktoken.py` | `mismatches: 0` |
| Perplexity matches NumPy | `uv run --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt` vs `./gpt2t_bin --ppl tests/data/alice_ch1.txt` | same to ~6 digits (25.2843) |
| Our quantizer matches NumPy | add `--quant 4,32,0` (bits, group, symmetric) to the reference; compare with `--dtype int4-g32` | same to ~6 digits |
| GGUF dequantization matches llama.cpp's Python `gguf` | `uv run --with gguf python tests/gguf_check.py gpt2/gguf/gpt2.Q4_K_M.gguf` (runs `tests/gguf_dump.mojo`) | relative error ≤ 1e-5 (measured ~4e-8 for all six block formats) |
| Integer (A16) kernels match exact math | `uv run mojo run -I . tests/a16_kernels.mojo` | `PASS` (max relative error ≤ 1e-5; measured ~1.6e-7) |
| A16 perplexity matches NumPy | `uv run --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt --quant 4,32,0 --quant-head 8,0,0 --act16` vs `./gpt2t_bin --dtype int4-g32-a16 --ppl tests/data/alice_ch1.txt` | same to ~1e-4 relative (int16 rounding flips; see M4) |
| GGUF perplexity matches NumPy | `uv run --with gguf --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt --gguf FILE` vs `./gpt2t_bin --gguf FILE --ppl tests/data/alice_ch1.txt` | same to ~6 digits |

Build both binaries before running these. Run `tests/same_as_baseline.sh`
after any change to `gpt2t.mojo`, `kernels.mojo`, or `tensor.mojo`: float32
must stay bit-identical. Lossy formats are checked with the accuracy harness
instead.

### Accuracy harness

```sh
./gpt2t_bin --dtype FMT --ppl tests/data/alice_ch1.txt --compare
```

Evaluation text: `tests/data/alice_ch1.txt`, Chapter I of *Alice's
Adventures in Wonderland* (public domain; Project Gutenberg eBook #11 with
the Gutenberg header and license removed), 3,308 GPT-2 tokens.

The text is scored in sliding windows of 1024 tokens starting every 512, and
each token is scored once, in the first window that contains its prediction,
so every scored token after the first window has at least 512 tokens of
context. `--compare` loads a float32 model too, runs both on the same
windows, and reports, over the 3,307 scored positions:

| Metric | Meaning |
|---|---|
| perplexity | exp(mean negative log-likelihood of the actual next token); lower is better. Reported for FMT and float32, with the % change |
| top-1 agreement | % of positions where FMT and float32 predict the same most-likely token |
| mean KL(f32 ‖ FMT) | how far FMT's whole predicted distribution moves from float32's, in nats; the most sensitive of the four (llama.cpp uses it for the same purpose) |
| logit \|diff\| | mean and max absolute difference of raw logits, over all positions and the whole vocabulary |

A run takes ~25 s without `--compare` and ~45 s with it (two models, 6
windows each, plus scoring 3,307 × 50,257 logits).

## Design

### Weight formats are a trait

Mojo has no class inheritance. A **trait** is an interface, and code generic
over it is compiled separately for each concrete type with its methods
inlined, so the abstraction costs nothing in the inner loops (M1 measured it:
the generic float32 build is as fast as the baseline).

The current trait, the formats' storage layouts (including the int4 packing
and the group-per-lane decode layout) and the kernel dispatch are described
in [README.md](README.md#tensor-structure); `tensor.mojo` is the source of
truth. What follows in this section is the original M1 design.

- Values are **handles** (a pointer plus shape). Copying copies the handle;
  `Model` owns them and calls `free` once in its destructor.
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

### M2. Accuracy harness ✅

Lossy formats need a number to judge them by, before they exist, so a
quantization bug can be told apart from quantization error.

- `--ppl FILE` and `--compare`, described under
  [Accuracy harness](#accuracy-harness).
- `Model.forward_all` computes logits at every position (`lm_head_rows` in
  kernels.mojo); `Model.blocks` is the shared transformer part, so the
  generation path is unchanged (still bit-identical).
- Evaluation text committed as `tests/data/alice_ch1.txt`.
- `tests/reference.py ppl` computes the same windows in NumPy.
- Acceptance (~4 significant digits vs NumPy): met with 6. Mojo 25.284337,
  NumPy 25.284352. float32 compared with itself gives 100% agreement and
  exactly zero KL and logit difference.

Possible later improvement: cache float32's logits (or per-position
statistics) so `--compare` doesn't recompute the baseline for every format
in a sweep.

### M3. 16-bit, int8/int4, and GGUF weights (float32 compute)

**Status and findings so far:**

- f16 / bf16: done (see [Results](#results)).
- Our own int8/int4 (`QuantMatrix`, `--dtype int...`): standard affine
  quantization, `w = scale * (u - zero_point)`, with a float16 `scale` and
  an **integer** `zero_point` (uint8, in the codes' range) per group, so 0 is
  exactly representable. (A first version stored a float16 offset, `w =
  scale * u + min`, like llama.cpp's Q4_1; replaced 2026-09-28 because the
  integer form is the standard definition and keeps the M4 kernel's inner
  sums in integers.) Verified against NumPy (`tests/reference.py --quant`)
  to 5-6 digits.
- **Output head at int8** (2026-09-28): `Model[W, E]` has a separate format
  E for wte (the tied embedding and output head). For int layer formats it
  defaults to `QuantMatrix[8, 0, False]` (int8, one scale and zero point per
  vocabulary row); `--head same` quantizes it like the layers. This took
  int4 group 32 from perplexity 352.6 to **25.59 (+1.2%)**, verified against
  NumPy (`--quant 4,32,0 --quant-head 8,0,0`: 25.592098 vs 25.592096).
  Asymmetric beats symmetric at every group size, and smaller groups are
  better (see Results). By KL, int4-g32 + int8 head (0.147) sits between
  GGUF Q4_0 (0.170) and Q4_K_M (0.098).
- **Fast kernels for our formats** (2026-09-28): `QuantMatrix` now stores
  `[OUT, IN]` (layer matrices are transposed in `from_f32`), so each row's
  groups are contiguous and it takes the OUT_MAJOR path (`matmul_rows`: fused
  dot for decode, float32 tiles for prefill). int4 is packed per 32 weights
  like Q4_0 (byte j = weights j and j+16). `dot_row` sums x·(u − zero_point)
  per group and applies the scale once per group. int4-g32 decode went from
  ~25 to ~99 tok/s, matching f16 at 37% of its size; int8-ch is the fastest
  format at ~124 tok/s. Perplexities still match NumPy to 6 digits and
  greedy decoding matches exactly. A trait member `FROM_GGUF` now marks
  GGUFMatrix (OUT_MAJOR no longer implies GGUF).
- Why int4 first failed (before the int8 head): we quantized the tied `wte`
  together with everything else, while llama.cpp keeps GPT-2's output head
  at 6-bit (Q6_K). GGUF Q4_0, whose layer matrices use nearly our
  `int4-g32-sym` scheme, reaches 27.2 with its 6-bit head.
- **GGUF** (`--gguf FILE`, gguf.mojo): pre-quantized GPT-2 124M files from
  Hugging Face, used as stored. GGUF keeps matrices as `[OUT, IN]` with
  blocks along the input axis, so they use `matmul_rows` (dequantize a row,
  then dot it with every token). The format is chosen per tensor at runtime
  because Q4_K_M mixes Q4_K, Q5_K and Q6_K. Dequantization and perplexity
  match llama.cpp's Python package and NumPy. **Q4_K_M: perplexity +0.7%.**
- GGUF kernels (gguf.mojo, kernels.matmul_rows):
  - Dequantizers are SIMD, 16 weights per step (masks and shifts on byte
    vectors, one conversion to float32). Each is compiled in two variants,
    store-to-buffer and fused-dot, differing only in `put`.
  - Decode (one token): the dot product is computed while dequantizing
    (`dot_row`), never writing the float32 weights. 1.3–2.5× faster per
    weight than dequantize-then-dot (research/test_dequant.mojo).
  - Prefill: 32 output rows at a time are dequantized into a float32 tile
    transposed to `[IN, 32]`, and the float32 tile kernel `mm_tile` runs over
    all tokens, so each weight is dequantized once per prompt.
  - Result: Q4_K_M decodes faster than f16 (107 vs 88 tok/s) and reads
    prompts faster than f32 (690 vs 559 tok/s). Decode is still arithmetic-
    bound (Q8_0 at 167 MB is slower than Q4_K_M at 105 MB); the output head
    (Q6_K, 38.6M weights per token) is the biggest single cost.
- Benchmarks: `tests/bench.sh [ROUNDS] FORMAT...` interleaves formats and
  prints medians.

**Original plan for our own formats (kept for reference):**

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

**Status (2026-09-28): implemented and verified.**

- `--dtype int4-g32-a16`, `int4-g64-a16`, `int4-g128-a16`, `int4-ch-a16`,
  `int4-g32-sym-a16`, `int8-ch-a16`: `QuantMatrix[..., A16=True]` with the
  output head as `QuantMatrix[8, 0, False, True]` (int8, integer path too).
- `kernels.quantize_rows` quantizes each matmul input row to int16 (symmetric,
  scale = max|x| / 32767). `kernels.matmul_rows_a16`: decode uses
  `QuantMatrix.dot_row_i16` (unpack 32 codes to int16 minus the zero point,
  one VPDPWSSD with 32 activations, per group int32 → float × scale); prefill
  unpacks each weight row to int16 once and reuses it for 4 tokens at a time
  (`dot4_i16`). `tensor.dot_pairs` wraps VPDPWSSD with a portable fallback.
- Trait members `ACT16`, `dot_row_i16`, `unpack_row_i16`, `group_size` have
  default bodies in the trait (abort), so only QuantMatrix implements them.
- Accuracy: int16 activations cost essentially nothing: int4-g32 25.5921
  (W4A32) → 25.5910 (W4A16). Verified against NumPy (`tests/reference.py
  --act16`): 25.591020 vs 25.590796 (5 digits; the remaining difference is
  rounding of exact .5 ties), and greedy decoding matches exactly.
- Speed (decode, short context, battery, interleaved): the gain grows with
  group size, because each group costs an int32 → float conversion and a
  multiply by its scale, and with 32-weight groups that equals the one
  VPDPWSSD per group:

  | Layers | float compute | int16 activations | gain |
  |---|---|---|---|
  | int4, group 32 | 95 | 87 | none |
  | int4, group 128 | 115 | 126 | +10% |
  | int4, per-channel | 119 | 138 | +16% |
  | int8, per-channel | 116 | 152 | +30% |

- **Integer prefill tiles** (`kernels.tile_i16`): each task unpacks 32
  output rows into the VNNI layout (for each input pair (k, k+1), the 32
  outputs' int16 weight pairs, so one 32-lane load covers 16 outputs × 2
  inputs), then runs register tiles of 4 tokens × 32 outputs: each
  activation pair is broadcast as one int32 and one VPDPWSSD per 16 outputs
  accumulates it. Group scales are applied once per group per tile, so even
  32-weight groups gain. One int32 lane sums a whole group here, so sums are
  flushed to float every 256 inputs (256 × 32767 × 255 < 2^31). Leftover
  rows (the vocabulary's 50257 % 32) use the row path (`rows_i16`).
  Prefill: int4-g32-a16 885 tok/s, int8-ch-a16 899 (float compute: 679 and
  624; f16: 538), on AC.
- `tests/a16_kernels.mojo` checks all integer paths (one token, tiles,
  leftover rows, token counts not a multiple of 4, and worst-case
  activations that would overflow int32 without flushing) against exact
  float64 math on the same quantized values: max error ~1.6e-7 relative.
  Perplexity differs between integer paths by ~1e-4 relative even though
  the kernels are exact: int16 rounding of activations turns 1e-7
  differences into occasional one-step flips, which compound over layers.
- **Group-per-lane decode layout for int4-g32 A16** (`QuantMatrix.BLOCKED`):
  with 32-weight groups each group used to fill a register, so every 32
  weights needed their own conversion and scale. Now every 256 weights (8
  groups) are stored as 8 steps of 16 bytes where lane i holds half of group
  i % 8, so after 8 VPDPWSSDs each lane has a half-group sum, and zero
  points, conversion and scales are applied once per 256 weights. The zero
  point is applied as sum(x*u) - zero_point*sum(x), with per-lane activation
  sums and the matching activation order computed once per token
  (`permute_x_i16`). Prefill unpacks this layout with SIMD plus one int32
  store per lane pair. int4-g32-a16 decode: 95 / 78 → **147 / 114 tok/s**
  (clean run, AC). Kernel tests pass (max error 3e-7), perplexity and greedy
  decode unchanged. Prefill after the SIMD unpack, re-measured on a quieter
  machine (load ~1-2, AC, 3 rounds): int4-g32-a16 803 tok/s (594-867),
  int4-ch-a16 851, f16 545; decode 140 / 105 vs 113 / 76 and 79 / 67.

Original plan:

Note: `QuantMatrix` already uses an integer zero point, so per group
`sum(x * w) = scale * (sum(x_q * u) - zero_point * sum(x_q))`, with both inner
sums in int32. With symmetric formats `zero_point` is the constant 8 (int4).

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

Perplexity, top-1 agreement, and KL are on `tests/data/alice_ch1.txt` (see
[Accuracy harness](#accuracy-harness)). GGUF speeds are medians of 3 rounds
of `tests/bench.sh`, measured together with f32 (559 / 60 / 45) and f16
(562 / 88 / 73) in the same session. *Our int formats: medians of 3 rounds
on battery, measured with f16 (561 / 100 / 85), Q4_K_M (608 / 108 / 89) and
Q4_0 (605 / 92 / 79) in the same session. †W4A16/W8A16: 3 rounds on
battery with f16 (624 / 94 / 78), int4-g32 (646 / 95 / 68), int8-ch
(572 / 116 / 96), Q4_K_M (570 / 84 / 64). ‡2 rounds on battery with int4-g128
(622 / 115 / 94) and int4-ch (607 / 119 / 97). §With the integer prefill
tiles: 3 rounds on AC with f16 (538 / 84 / 58), int4-g32 (679 / 85 / 68),
int8-ch (624 / 112 / 79). int4-g128-a16's prefill was measured before the
tiles. ¶With the group-per-lane decode layout: 3 rounds on AC with int4-g32
(595 / 88 / 74), int4-ch-a16 (920 / 130 / 102), int8-ch-a16 (905 / 138 /
113), f16 (546 / 90 / 38). Prefill 803 is from a later 3-round re-run with
f16 (545 / 79 / 67) and int4-ch-a16 (851 / 113 / 76). Speed ranges are from two interleaved
rounds; the machine's speed drifts by ±15% between runs (thermal), so compare
formats measured in the same session.

Notes on f16 / bf16:
- bf16's *lower* perplexity is luck, not better accuracy: its KL is ~430×
  f16's and it changes the top prediction at ~5% of positions. f16 is
  essentially lossless. This is why the harness reports KL, not just
  perplexity.
- Decode is ~1.3–1.6× faster than f32, less than the ~1.85× the streaming
  probe predicted: per-token costs that don't shrink with the weights (thread
  wake-ups for ~60 parallel regions, attention, the KV cache) are a larger
  share of each step.
- f16 is slower than bf16 in both prefill and decode: VCVTPH2PS costs more
  than bf16's shift. Load time is higher for 16-bit formats because of the
  conversion (bf16 ~1.2 s: its rounding is done in software).

| Format | Weights | Prefill tok/s | Decode tok/s (short / long ctx) | Perplexity | Top-1 vs f32 | Mean KL vs f32 |
|---|---|---|---|---|---|---|
| f32 (baseline `gpt2.mojo`) | 474 MB | ~620 | ~62 / ~57 | 25.284 | 100% | 0 |
| f32 (`gpt2t`, 2026-09-27 evening run) | 474 MB | 546–631 | 48–64 / 42–45 | 25.284 | 100% | 0 |
| f32 (`gpt2t`) | 474 MB | same | same | 25.284 | 100% (bit-identical) | 0 |
| f16 | 239 MB | 504–562 | 64–77 / 59–65 | 25.302 (+0.07%) | 99.67% | 1.7e-5 |
| bf16 | 239 MB | 603–613 | 78–82 / 67–69 | 25.073 (−0.84%) | 95.13% | 7.4e-3 |
| int8, per-channel (ours, head int8) | 121 MB | 660* | 124 / 101* | 26.116 (+3.29%) | 81.6% | 4.6e-2 |
| **int4, group 32 + int8 head (ours)** | 88 MB | 642* | 99 / 85* | **25.592 (+1.22%)** | 75.1% | 1.5e-1 |
| int4, group 64 + int8 head (ours) | 84 MB | 668* | 111 / 92* | 27.560 (+9.0%) | | |
| int4, group 128 + int8 head (ours) | | | | 27.672 (+9.4%) | | |
| int4, per-channel + int8 head (ours) | | | | 28.821 (+14.0%) | | |
| int4, group 32 sym + int8 head (ours) | | | | 29.806 (+17.9%) | | |
| int4, group 64 sym + int8 head (ours) | | | | 29.863 (+18.1%) | | |
| int4, group 128 sym + int8 head (ours) | | | | 32.029 (+26.7%) | | |
| int4, per-channel sym + int8 head (ours) | | | | 51.819 (+105%) | | |
| int4, group 32, head int4 too (`--head same`) | | | | 352.6 | | |
| int4, group 32 sym, head int4 too | | | | 748.7 | | |
| GGUF Q8_0 (head Q6_K) | 167 MB | 574 | 63 / 56 | 25.416 (+0.52%) | 91.1% | 1.2e-2 |
| GGUF Q4_0 (head Q6_K) | 99 MB | 710 | 80 / 79 | 27.178 (+7.49%) | 74.6% | 1.7e-1 |
| GGUF Q4_K_M (Q4_K/Q5_K/Q6_K, head Q6_K) | 105 MB | 690 | 107 / 90 | 25.460 (+0.70%) | 80.2% | 9.8e-2 |
| GGUF i1-Q4_K_M (NumPy only so far) | 105 MB | | | 27.615 (+9.2%) | | |
| **int4, group 32, W4A16 + int8 head (A16)** | 88 MB | 803¶ | **147 / 114**¶ (140 / 105 in the re-run) | 25.593 (+1.22%) | | |
| int8, per-channel, W8A16 | 121 MB | **899**§ | 135 / 103§ (152 / 119†) | 26.117 (+3.29%) | | |
| int4, group 128, W4A16 | 82 MB | 469‡ | 126 / 105‡ | | | |
| int4, per-channel, W4A16 | 81 MB | 888§ | 126 / 100§ | | | |

## Open questions

- Q4_K_M changes perplexity by only 0.7% but agrees with float32's top-1 at
  only 80% of positions (KL 0.098), and even Q8_0 agrees at only 91%. GPT-2
  small is known to be sensitive to quantization; is top-1 agreement on this
  text dominated by near-ties? A second evaluation text would help.
- The imatrix file (i1-Q4_K_M) scores worse than plain Q4_K_M here (+9.2% vs
  +0.7%). Probably its importance data came from different text; one 3.3K
  token text can't settle it.
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
