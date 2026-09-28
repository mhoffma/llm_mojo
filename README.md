# llm_mojo

GPT-2 124M inference on the CPU, written from scratch in [Mojo](https://www.modular.com/mojo) 1.1, with pluggable weight formats: float32, float16, bfloat16, our own affine int8/int4 quantization, llama.cpp GGUF files, and integer W4A16 / W8A16 kernels on AVX-512 VNNI.

The project measures **what each weight format costs in accuracy and gains in speed** on a laptop CPU. Every format runs through the same model code and the same accuracy test against float32. Every kernel is checked against an independent NumPy implementation.

int4 weights (group 32) with an int8 output head and int16 activations: 88 MB instead of 474 MB, about +1% perplexity, and 3.2× float32's decode speed (300 vs 93 tok/s) on a 4-core laptop:

```text
$ MODULAR_THREAD_BUSY_WAIT_US=0 ./gpt2t_bin --dtype int4-g32-a16 -n 40 "In a shocking finding, scientists discovered a herd of unicorns"
In a shocking finding, scientists discovered a herd of unicorns, a feat that could not be achieved without the help of a little help from a tiny, non-human centipede!

Researchers from the University of Copenhagen found the tiny unicorns,
---
dtype int4-g32-a16+head-int8-ch-a16+kv-int8 | 88 MB | load 2006 ms | prompt 12 tokens in 55 ms ( 217 tok/s ) | generated 40 tokens in 144 ms ( 277 tok/s )
```

(Generation uses top-k 40 sampling with temperature 0.8 and a fixed seed, so the text repeats run to run. The load time includes quantizing the float32 weights at startup.)

## Contents

1. [Key features](#key-features)
2. [Results](#results)
3. [Quick start](#quick-start)
4. [Command-line reference](#command-line-reference)
5. [Tensor structure](#tensor-structure)
6. [Kernels](#kernels)
7. [Measuring accuracy](#measuring-accuracy)
8. [Verification](#verification)
9. [Performance notes](#performance-notes)
10. [Project layout](#project-layout)
11. [Status and roadmap](#status-and-roadmap)
12. [Credits](#credits)

## Key features

- **Complete GPT-2 124M inference in Mojo, with no ML framework**:
  - a loader that reads Hugging Face `safetensors` and llama.cpp GGUF files directly
  - a byte-level BPE tokenizer that matches OpenAI's `tiktoken`
  - a KV cache, top-k / temperature sampling, and streaming output
- **Weight formats as a trait.** `WeightMatrix` is one interface with several implementations:
  - float32, float16 and bfloat16
  - our own affine int8 and int4, with a float16 scale and an integer zero point per group of 32, 64 or 128 weights, or per channel
  - llama.cpp GGUF Q4_0, Q4_1, Q8_0, Q4_K, Q5_K and Q6_K, used as stored

  The model is compiled separately for each format, so the abstraction costs nothing at run time. float32 through the generic code is bit-identical to the plain float32 program.
- **A mixed-precision output head.** The tied embedding and output head (`wte`) can use a different format from the layers. Keeping it at int8 while the layers are int4 is what makes int4 usable: perplexity drops from 352.6 to 25.59.
- **Integer inference (W4A16 / W8A16).** Activations are quantized to int16 at run time, and matmuls run as int16 × int16 → int32 with the AVX-512 VNNI `VPDPWSSD` instruction:
  - Decode (one token at a time) uses a custom group-per-lane code layout.
  - Prompts use a VNNI register-tiled kernel.
  - GGUF output heads (Q6_K, Q8_0) also run in integers: ~1.7x faster than dequantizing to float.
  - int4 group 32 with int16 activations and integer attention is our fastest decoder (~235 tok/s after a short prompt), at +1.2% perplexity and 88 MB.
- **Fast kernels for every format:**
  - SIMD dequantization with the dot product fused in, for decode
  - dequantize-once float32 tiles, or integer VNNI tiles, for prompt processing
  - multithreading tuned for a 4-core laptop: each decoded token runs as one parallel region, with a team of threads (one per core) that meet at a barrier between operations
- **An accuracy harness.** `--ppl FILE --compare` measures perplexity with sliding windows, top-1 agreement with float32, KL divergence from float32, and logit differences.
- **Verification at every level:**
  - a NumPy GPT-2 reference that also implements every quantization scheme
  - a dequantization check against llama.cpp's Python `gguf` package
  - kernel unit tests against exact float64 arithmetic
  - a bit-exactness test against the float32 baseline
- **Profiling built in.** `--profile` prints the time per token of each operation (every matmul, attention, KV store, output head) for prompt and decode tokens.
- **Learning material.** `research/` holds small, commented Mojo programs, each answering one question (VNNI from Mojo, float16/bfloat16 support, dequantization speed, integer softmax, spin barriers vs parallel regions). `PLAN.md` documents every decision, measurement and Mojo 1.1 pitfall met along the way.

## Results

All measurements are on an Intel i7-1160G7 laptop (Tiger Lake, 4 cores / 8 threads, AVX-512 + VNNI, 16 GB), on AC power, with `MODULAR_THREAD_BUSY_WAIT_US=0`.

- **Accuracy** is measured on `tests/data/alice_ch1.txt` (3,307 scored tokens) against float32. Lower perplexity is better; lower KL means closer to float32.
- **Prompt speed** is for a 476-token prompt. **Decode speed** is for 100 generated tokens after a 1-token prompt (short) and after the 476-token prompt (long).
- Accuracy and speed are from one clean session (2026-09-28, after the M6 tuning): speeds are medians of 5 interleaved rounds of `tests/bench.sh`, which varied by only a few percent between rounds. The frozen float32 program `gpt2.mojo` ran at 635 / 63 / 56 tok/s in the same session. Older, session-by-session numbers are in [`PLAN.md`](PLAN.md#results).
- Quantized formats use an int8 KV cache by default (f16 and bf16 keep their own type), so their accuracy includes the cache's error.

| Format | Weights | Perplexity (Δ vs f32) | Top-1 vs f32 | Mean KL vs f32 | Prompt tok/s | Decode tok/s (short / long) |
|---|---|---|---|---|---|---|
| f32 | 474 MB | 25.284 | 100% | 0 | 656 | 93 / 83 |
| f16 | 239 MB | 25.303 (+0.07%) | 99.7% | 0.00002 | 809 | 157 / 139 |
| bf16 | 239 MB | 25.065 (−0.87%)¹ | 95.2% | 0.0075 | 760 | 156 / 138 |
| GGUF Q8_0 (8-bit head) | 167 MB | 25.431 (+0.58%) | 91.1% | 0.012 | 881 | 128 / 118 |
| GGUF Q4_K_M (4/5/6-bit mix, 6-bit head) | 105 MB | 25.475 (+0.75%) | 80.0% | 0.098 | 873 | 192 / 171 |
| GGUF Q4_0 (6-bit head) | 99 MB | 27.197 (+7.6%) | 74.4% | 0.169 | 878 | 138 / 127 |
| int8 per-channel (ours, `int8-ch`) | 121 MB | 26.142 (+3.4%) | 81.6% | 0.046 | 857 | 200 / 175 |
| int4 group 32, int8 head (ours, `int4-g32`) | 88 MB | 25.521 (+0.93%) | 75.1% | 0.148 | 883 | 150 / 136 |
| int8 per-channel, int16 activations (`int8-ch-a16`) | 121 MB | 26.164 (+3.5%) | 81.6% | 0.046 | **1358** | 283 / 241 |
| int4 group 32, int8 head, int16 activations (`int4-g32-a16`) | 88 MB | 25.553 (+1.1%) | 75.2% | 0.148 | 1146 | 300 / 253 |
| **`int4-g32-a16 --attention int`** (integer attention) | **88 MB** | **25.536 (+0.99%)** | 75.5% | 0.148 | 1150 | **300 / 278** |
| int4 per-channel, int16 activations (`int4-ch-a16`) | 81 MB | 28.878 (+14.2%) | 61.0% | 0.408 | 1310 | 251 / 214 |

¹ bfloat16's lower perplexity is chance on this text, not better accuracy. Its KL divergence is about 430× float16's, and it changes the top prediction at 5% of positions.

**What the numbers show:**

- **float16 is essentially lossless** at half the size. bfloat16 is clearly less accurate for GPT-2's weights; its advantage is a wider range, which these weights don't need.
- **The output head is the tensor that can't go to 4 bits.** With the head quantized to int4 too, int4 group 32 reaches perplexity 352.6. With an int8 head it's 25.59. llama.cpp keeps GPT-2's head at 6-bit for the same reason.
- **Small groups matter at 4 bits:**
  - group 32: +1.2%
  - group 64: +9.0%
  - group 128: +9.4%
  - per-channel: +14%
  - symmetric variants: +18% to +105%

  Asymmetric quantization (with a zero point) beats symmetric at every group size.
- **Our simple round-to-nearest int4 (group 32, int8 head) lands between llama.cpp's Q4_0 and Q4_K_M.** It's better than Q4_0 and worse than Q4_K_M, which uses error-minimizing scales and 5/6-bit layers, and it's the smallest of the three.
- **Perplexity alone is misleading.** Q4_K_M changes perplexity by only 0.7% but changes the top prediction at 20% of positions. The harness reports KL divergence for this reason.
- **int16 activations cost almost nothing in accuracy** (int4-g32: 25.592 → 25.591 with a float32 cache; 25.521 → 25.553 with the int8 cache, 0.1%) **and make the integer kernels faster**:
  - Decode: int4-g32 150 → 300 tok/s, int8-ch 200 → 283 tok/s.
  - Prompts: 30–60% faster than the same weights with float compute (int8-ch 857 → 1358 tok/s).
- **Integer attention is free and helps long contexts:** same accuracy, decode after 476 tokens 253 → 278 tok/s.
- **Decode speed follows bytes per token, once the per-token overheads are gone.** float32 → int4-a16 is 5.4× fewer weight bytes and 3.2× faster decode. The M6 tuning (one parallel region per token, vectorized epilogues, faster output heads) roughly doubled decode for every format since the first results (f16 ~90 → 157, int4-g32-a16 ~140 → 300 tok/s).
- **GGUF files run as stored, without re-quantizing,** but their float dequantize-and-multiply kernels are slower than our integer ones (Q4_K_M 192 vs int4-g32-a16 300 tok/s decode).

(The group-size sweep above was measured with a float32 KV cache.)

## Quick start

### Requirements

- **Linux on x86-64.**
  - Everything runs on any AVX2 CPU.
  - The integer `-a16` formats are fast only with AVX-512 VNNI or AVX-VNNI; without it they fall back to much slower portable code. Check with `grep -o 'avx512_vnni\|avx_vnni' /proc/cpuinfo | sort -u`.
- **[uv](https://docs.astral.sh/uv/)**, the Python package manager, which installs Mojo.
- **About 2.5 GB of disk:**
  - ~1.1 GB for the environment
  - ~550 MB for the GPT-2 weights
  - ~600 MB if you also download the GGUF files
- **Internet access** for the first setup.

### Setup

```sh
git clone git@github.com:mhoffma/llm_mojo.git
cd llm_mojo
uv sync                     # .venv with mojo==1.1.0 and max==26.6.0
uv run mojo --version       # Mojo 1.1.0

# GPT-2 124M weights and tokenizer (Hugging Face, ~550 MB)
mkdir -p gpt2
for f in model.safetensors vocab.json merges.txt; do
  curl -L -o gpt2/$f https://huggingface.co/openai-community/gpt2/resolve/main/$f
done

# Optional: pre-quantized llama.cpp GGUF files of the same model
mkdir -p gpt2/gguf
curl -L -o gpt2/gguf/gpt2.Q4_K_M.gguf https://huggingface.co/mradermacher/gpt2-GGUF/resolve/main/gpt2.Q4_K_M.gguf
for q in Q4_0 Q4_1 Q8_0; do
  curl -L -o gpt2/gguf/gpt2.$q.gguf https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.$q.gguf
done
```

The `max` package is needed because in Mojo 1.1, `parallelize` lives in `max.algorithm` rather than the standard library. The two versions must match: `max 26.6.0` pairs with `mojo 1.1.0`.

### Build and run

```sh
uv run mojo build -o gpt2t_bin gpt2t.mojo   # ~30 s: compiles every format
export MODULAR_THREAD_BUSY_WAIT_US=0        # see Performance notes

# Generate text
./gpt2t_bin -n 100 "In a shocking finding, scientists discovered"          # float32
./gpt2t_bin --dtype int4-g32-a16 -n 100 "The meaning of life is"         # int4 weights, int16 activations
./gpt2t_bin --gguf gpt2/gguf/gpt2.Q4_K_M.gguf -t 0 -n 50 "Hello, my name is"   # llama.cpp weights, greedy

# Measure accuracy against float32 (~45 s)
./gpt2t_bin --dtype int4-g32 --ppl tests/data/alice_ch1.txt --compare

# Compare speed (interleaved rounds, medians)
tests/bench.sh 3 f16 int4-g32-a16 gpt2/gguf/gpt2.Q4_K_M.gguf
```

Timing and model size are printed to stderr after the generated text.

`gpt2.mojo` is the original single-file float32 program. It's kept unchanged as the reference that `gpt2t` is tested against (`uv run mojo build -o gpt2_bin gpt2.mojo`).

## Command-line reference

`./gpt2t_bin [options] "prompt"`

| Option | Meaning | Default |
|---|---|---|
| `--dtype FMT` | weight format of the layer matrices (see below) | `f32` |
| `--head FMT` | with an int `--dtype`: format of `wte` (embedding + output head), `int8` or `same` | `int8` |
| `--kv FMT` | KV cache format: `auto` (f32/f16/bf16 weights keep their type, quantized weights get int8), `f32`, `f16`, `bf16`, `int16`, `int8` (see [KV cache](#the-kv-cache)) | `auto` |
| `--attention A` | `float`, or `int`: attention computed in integers (needs the int8 cache; see [integer attention](#integer-attention)) | `float` |
| `--gguf FILE` | use a llama.cpp GGUF file of GPT-2 124M, as stored; overrides `--dtype` | |
| `-m DIR` | directory with `model.safetensors`, `vocab.json`, `merges.txt` | `gpt2` |
| `-n N` | tokens to generate | 64 |
| `-t TEMP` | sampling temperature; `0` = greedy | 0.8 |
| `-k K` | sample from the K most likely tokens; `0` = all | 40 |
| `-s SEED` | random seed | 1337 |
| `-v` | print the prompt's token ids and the top-5 next-token logits | off |
| `--ppl FILE` | measure perplexity on a text file instead of generating | |
| `--compare` | with `--ppl`: also run float32 and report agreement, KL, logit differences | off |
| `--profile` | print time per token by operation (each matmul, attention, KV store, output head, …), for decode and prompt tokens | off |
| `--threads N` | threads that decode each token together (see [Decode with a thread team](#decode-with-a-thread-team)) | one per core (half the runtime's threads) |

**`--dtype` formats:**

| Name | Layer weights | Compute |
|---|---|---|
| `f32`, `f16`, `bf16` | dense floats | float32 |
| `int8-ch` | int8, one scale + zero point per output channel | float32 |
| `int4-ch`, `int4-g128`, `int4-g64`, `int4-g32` | int4, per channel or per group of 128 / 64 / 32 | float32 |
| any int format + `-sym` (e.g. `int4-g32-sym`) | symmetric (zero point fixed at the middle code) | float32 |
| `int8-ch-a16`, `int4-ch-a16`, `int4-g128-a16`, `int4-g64-a16`, `int4-g32-a16`, `int4-g32-sym-a16` | as above | int16 activations, integer VNNI |

With int formats, the head defaults to int8 per vocabulary row (`--head int8`), using the integer path for `-a16` formats. `--head same` quantizes it like the layers.

## Tensor structure

### The model's tensors

GPT-2 124M has 12 transformer blocks, hidden size C = 768, 12 attention heads of 64 dimensions, a 50,257-token vocabulary and a 1,024-token context.

| Tensor | Shape (Hugging Face) | Stored as |
|---|---|---|
| `wte`: token embedding, tied to the output head | [50257, 768] | format **E** (`Model[W, E]`) |
| per block: `attn.c_attn` (Q, K, V) | [768, 2304] | format **W** |
| per block: `attn.c_proj` | [768, 768] | format **W** |
| per block: `mlp.c_fc` | [768, 3072] | format **W** |
| per block: `mlp.c_proj` | [3072, 768] | format **W** |
| `wpe`: position embedding, LayerNorm weights and biases, all biases | small | float32, in one buffer |

The five large matrices are about 99% of the bytes read per token; `wte` alone is about 30%, because it's also the output head. `Model[W, E]` takes the layer format `W` and the embedding/head format `E`, which defaults to `W`. GGUF files store a separate, untied output head, which the model holds as its own field (`lm`).

Activations, the residual stream and attention arithmetic are float32. The `-a16` formats quantize only the *inputs* of the matmuls to int16. The KV cache has its own format; see [the KV cache](#the-kv-cache).

### The KV cache

For every layer, the cache holds each past token's key and value vectors, `[layer, position, head, 64]`. Each new token's attention compares its query with every cached key, and adds up the cached values weighted by the softmax of those scores. In float32 that's 72 KB per token and 75.5 MB for a full 1,024-token context. Once the weights are small, reading the cache is a large share of long-context decoding: at position 476, int4-g32-a16 reads ~87 MB of weights plus 35 MB of float32 cache per token.

Like the weights, the cache format is a trait, `KVCache` in `kvcache.mojo`. The attention kernel is generic over it, and it's the model's third format parameter, `Model[W, E, KV]`:

| `--kv` | Stores | Per token |
|---|---|---|
| `f32` | float32 | 72 KB |
| `f16`, `bf16` | 16-bit floats, widened when read | 36 KB |
| `int16`, `int8` | symmetric integers, one float32 scale per (layer, position, head) for keys and one for values | ~37 KB, ~19 KB |

- **The default, `auto`, matches the model's precision.** f32, f16 and bf16 weights keep their own type, and quantized weights (ours and GGUF) get int8, chosen from the measurements below. `--kv` overrides it.
- **For the integer caches:**
  - `store` quantizes each head's 64 values when a token is added.
  - `score` applies the key's scale once per dot product.
  - `add_value` folds the value's scale into the softmax weight, so there's no per-element scaling.

**Accuracy** (measured with `tests/compare.sh`; details in [`PLAN.md`](PLAN.md#m5-pluggable-kv-cache-formats)):
- **The cache alone, with float32 weights:**
  - int16: KL 6×10⁻⁹
  - f16: KL 5×10⁻⁷
  - bf16: KL 3×10⁻⁵
  - int8: KL 3×10⁻⁴, +0.09% perplexity, 99.0% top-1 agreement
- **With quantized weights,** an int16 cache changes nothing, and an int8 cache adds at most ~0.001 KL on top of the weights' 0.05–0.15. The cache is not where the error comes from.

These results are verified against NumPy (`tests/reference.py --kv`).

**Speed** (int4-g32-a16 weights, tok/s; 3 interleaved rounds of `tests/bench.sh`, with `LONG_REPEAT=53` for the ~900-token prompt):

| KV cache | Prompt, 476 tokens | Decode after 476 | Prompt, ~900 tokens | Decode after ~900 |
|---|---|---|---|---|
| f32 | 879 | 116 | 690 | 100 |
| int16 | 938 | 121 | 783 | 106 |
| int8 | 975 | 128 | 847 | 116 |

The smaller the cache, the less memory attention reads per token, and the gain grows with the context: int8 is 16% faster than f32 for decode after ~900 tokens, and 23% faster for the prompt.

Note: the int and GGUF results in the tables above were measured with a float32 cache, before the int8 default existed. `--kv f32` reproduces them.

### Integer attention

Attention belongs to the cache format: `KVCache.attend(...)`. The model code and `kernels.attention` call it the same way for every format, and only the type the model is instantiated with differs:

- **Float caches** (`FloatKV`: DenseKV, QuantKV) share the original float32 attention, `kvcache.attend_float`.
- **`IntAttnKV`** (`int_attention.mojo`, selected with `--attention int` on the int8 cache) stores int8 keys and values in layouts built for VNNI and computes attention in integers. Per token and head:
  1. The query is quantized to int16.
  2. Keys are stored 16 positions per register, in pairs along the head dimension, so one broadcast query pair and one `VPDPWSSD` advance 16 positions' scores.
  3. Scores are rescaled with integer multipliers into one shared fixed-point scale, since each key has its own scale.
  4. `intmath.masked_exp`, the integer softmax without its normalization (`e^x = 2^(x·log2 e)`: a shift for the integer part, a degree-3 polynomial for the fraction), gives weights whose largest is exactly 1.0.
  5. Values are stored in position pairs, so one `VPDPWSSD` adds two positions' contributions to 16 dimensions.
  6. The output is divided by the sum of the weights once at the end, in the float multiply that converts the result for the next matmul.

  Normalizing at the end rather than first keeps the weights precise when attention is spread over many positions. The first version, which normalized to 16-bit probabilities first, was off by up to 1.8% over ~1,000 flat positions; this one is within 4×10⁻⁴.

**Cost and gain** (int4-g32-a16, int8 cache):
- **Accuracy:** unchanged (KL 0.148 either way). With float32 weights, integer attention adds 3×10⁻⁵ KL.
- **Speed:** decode is 9% faster after 476 tokens and 15% after ~900 (135 and 130 tok/s against 124 and 113). Prompt processing is 5–8% faster.

`intmath.masked_softmax` (with normalization, 16-bit output) is also available on its own. It's tested in `tests/int_softmax.mojo`, and integer attention in `tests/int_attention_check.mojo`.

### The `WeightMatrix` trait

Each weight format is a struct implementing one trait (`tensor.mojo`). The kernels are generic over it, and `main` turns the `--dtype` string into a compile-time type once (`run[QuantMatrix[4, 32, False, True], HEAD8_A16](args)`), so everything below that point is specialized and inlined for the format.

```mojo
trait WeightMatrix(Deinitable, ImplicitlyCopyable):
    comptime OUT_MAJOR: Bool   # stored [OUT, IN] (one output per row) instead of [IN, OUT]
    comptime FROM_GGUF: Bool   # read from a GGUF file as stored, not converted from float32
    comptime ACT16: Bool       # matmuls use int16 activations and integer kernels

    @staticmethod
    def name() -> String                                  # the --dtype name
    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int, reduce_rows: Bool) -> Self
    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]
    def dequant_row(self, row: Int, dst: FPtr)            # one row as float32
    def dot_row(self, row: Int, x: FPtr) -> Float32       # fused dequantize + dot
    def nbytes(self) -> Int
    def free(self)

    # Integer path (ACT16 formats); default bodies abort, so only QuantMatrix implements them
    def permute_x_i16(self, xq: I16Ptr, n: Int, dst: I16Ptr, sums: FPtr) -> Bool
    def dot_row_i16(self, row: Int, xq: I16Ptr, xsums: FPtr) -> Float32
    def unpack_row_i16(self, row: Int, dst: I16Ptr, scales: FPtr)
    def group_size(self) -> Int
```

- **Values are handles** (pointers plus shape). Copying one copies the handle; the model owns them and frees each once.
- **`reduce_rows`** tells `from_f32` which axis the dot products sum over: down the rows for layer matrices (`[IN, OUT]`), along the rows for `wte` (`[V, C]`). Quantization groups run along that axis.
- **Mojo has no class inheritance.** Traits plus compile-time generics give the same flexibility with no virtual-call cost. The trait's default method bodies spare the float-only formats from stubbing out the integer methods.

### Formats

#### `DenseMatrix[dtype]`: float32 / float16 / bfloat16

Stored in Hugging Face's `[IN, OUT]` layout. `load` widens to float32 with `.cast`:
- free for float32
- one `VCVTPH2PS` for float16 (F16C)
- a 16-bit shift for bfloat16

This CPU has no native 16-bit float arithmetic, so the 16-bit types are storage formats only (`research/test_half.mojo`).

#### `QuantMatrix[BITS, GROUP, SYMMETRIC, A16]`: our affine int8 / int4

Standard affine quantization, as in PyTorch, ONNX and TFLite:

```text
w = scale * (u - zero_point)

u           unsigned BITS-bit code          (0..15 for int4, 0..255 for int8)
scale       float16, one per group
zero_point  uint8,   one per group, in the codes' range, so w = 0 is exact
```

- **Groups:** GROUP consecutive weights along the reduction axis (32, 64 or 128), or a whole row when GROUP = 0 ("per-channel").
- **Asymmetric** (default): the group's range, widened to include 0, is mapped onto all codes. `scale = (max − min) / (2^BITS − 1)` and `zero_point = round(−min / scale)`.
- **Symmetric** (`-sym`): `zero_point = 2^(BITS−1)`, and `scale = max|w| / (2^(BITS−1) − 1)`.
- **Rounding** is plain round-to-nearest.
- **Storage cost:** 3 bytes per group. int4 with group 32 therefore costs 4.75 bits per weight.

**Layout.** Matrices are stored `[OUT, IN]`: layer matrices are transposed at load time, and `wte` is already `[V, C]`. Each output's weights, and so its quantization groups, are therefore contiguous. int4 codes are packed in 32-weight chunks the same way llama.cpp's Q4_0 does it, so one mask and one shift unpack 16 codes:

```text
32 weights -> 16 bytes
byte j (j = 0..15):  low nibble = weight j,  high nibble = weight j + 16

unpack:  (bytes & 0xF) -> weights 0..15      (bytes >> 4) -> weights 16..31
```

**Group-per-lane layout** (int4, group 32, `A16`; `QuantMatrix.BLOCKED`). With the layout above, each 32-weight group fills a whole register, so every 32 weights would need their own int32-to-float conversion and scale multiply. Instead, `int4-g32*-a16` stores every 256 weights (8 groups) as 8 "steps" of 16 bytes, arranged so that each int32 lane collects half of one group:

```text
block = 256 weights = groups g0..g7 (32 weights each), stored as steps s = 0..7 (16 bytes each)

unpacking step s gives 32 int16 codes = 16 lanes (pairs):
  lane L (0..15) holds codes 2p and 2p+1 of group L % 8,  where p = 8 * (L // 8) + s
  lanes 0-7:  first halves of g0..g7
  lanes 8-15: second halves of g0..g7

after VPDPWSSD over all 8 steps, lane L = sum over half a group of x * u
```

Zero points, the int32-to-float conversion and the scales are then applied once per 256 weights, each as one 16-lane vector operation. The zero point is applied as `Σ x·u − zero_point · Σ x`. The activations are reordered to match, and their per-lane sums computed, once per token and shared by every row (`permute_x_i16`).

#### `GGUFMatrix`: llama.cpp GGUF files

`gguf.mojo` reads the GGUF container (header, metadata, tensor index, aligned data) and uses each tensor as stored, in place. The format is a runtime property of each tensor, because a file like Q4_K_M mixes formats within one model.

| Type | Weights / block | Bytes | Encoding |
|---|---|---|---|
| Q4_0 | 32 | 18 | f16 `d`; 4-bit `q`; `w = d·(q − 8)` |
| Q4_1 | 32 | 20 | f16 `d`, `m`; 4-bit `q`; `w = d·q + m` |
| Q8_0 | 32 | 34 | f16 `d`; int8 `q`; `w = d·q` |
| Q4_K | 256 | 144 | f16 `d`, `dmin`; 6-bit scale and min for each of 8 sub-blocks; 4-bit `q` |
| Q5_K | 256 | 176 | as Q4_K, plus a 5th bit per weight |
| Q6_K | 256 | 210 | 4 + 2-bit `q`; int8 scale per 16 weights; f16 `d`; `w = d·sc·(q − 32)` |
| F16, F32 | 1 | 2, 4 | plain |

GGUF also stores matrices `[OUT, IN]`, with blocks along the input axis.

## Kernels

`kernels.linear` and `kernels.head` pick a kernel at compile time from the format's trait members:

| Format | Kernel | Decode (1 token) | Prompt (T tokens) |
|---|---|---|---|
| `DenseMatrix` (`[IN, OUT]`) | `matmul` | split-K GEMV: threads stream whole rows, partial sums reduced | 8 tokens × 32 outputs register tile (`mm_tile`) |
| `OUT_MAJOR` float compute: GGUF, `QuantMatrix` | `matmul_rows` | `dot_row`: SIMD dequantization with the dot product fused in, never writing float32 weights | dequantize 32 rows once into a transposed float32 tile, then `mm_tile` over all tokens |
| `ACT16`: `QuantMatrix[..., A16=True]` | `matmul_rows_a16` | `dot_row_i16`: unpack to int16, `VPDPWSSD`, group-per-lane layout for group 32 | VNNI register tile: 4 tokens × 32 outputs, weights in pair-interleaved layout (`tile_i16`) |

Details of the integer path:

- **Activations:** `quantize_rows` quantizes each input row to int16, symmetric, with `scale = max|x| / 32767`.
- **Multiply-adds:** `tensor.dot_pairs` wraps `VPDPWSSD` (`llvm.x86.avx512.vpdpwssd.512`), with a portable fallback for CPUs without VNNI.
- **Prompt tile layout:** the tile kernel stores weights in the standard VNNI layout. For each input pair (k, k+1), 16 outputs' weight pairs share one register. Each activation pair is broadcast as one int32, and one instruction advances 16 outputs.
- **Overflow:** in the tile kernel, one int32 lane accumulates a whole group, so the sums move to float every 256 inputs. 256 × 32767 × 255 < 2³¹, so int32 can't overflow even with int8 weights.

Other kernels:

- **LayerNorm:** SIMD.
- **Attention:** multi-head with a KV cache, computed by the cache format (`KVCache.attend_one`, per token and head). For prompts, it runs on one thread when there's little work, because waking threads costs more than the work.
- **GELU and residual adds** are fused into the matmul epilogues.

### Decode with a thread team

Profiling (`--profile`) showed every matmul call paying ~80 µs whatever its size: the cost of starting a parallel region, which wakes the worker threads and waits for all of them. With ~50 regions per token, that was ~40% of decode time. `research/test_barrier.mojo` measured a parallel region at ~23 µs of pure overhead, against ~1 µs for a spin barrier.

So a decoded token now runs as **one** parallel region (`Model.decode`):
- **A team of threads** (`team.mojo`) goes through every operation of the step together, each doing its share, and they meet at a barrier between operations.
- **Thread 0** does the small serial pieces: the embedding, LayerNorm, storing the token's key and value.
- **The shared operations:** the matmuls (`kernels.linear_team`), attention (heads divided among threads), and the output head (`head_team`).
- **Same arithmetic as before:** the partitioning and summation order match the region-per-operation kernels, so outputs are identical and float32 is still bit-identical.
- **Prompts keep one region per operation.** Their cost is spread over many tokens.

**Two things mattered for speed:**
- **One thread per physical core.** With all 8 hyperthreads of a 4-core CPU, waiting threads slow their sibling doing real work on the same core, and the team decode was slower than before. With 4 it's ~40% faster. The default is half the runtime's threads, and `--threads N` overrides it.
- **Spin, then yield.** The barrier spins briefly, then yields its CPU while waiting. On a busy machine, a team member can be descheduled, and pure spinning would then make the others wait for a whole scheduler time slice.

| Decode tok/s (short / long context) | One region per operation | Team, 8 threads | Team, 4 threads (default) |
|---|---|---|---|
| f16 | 100 / 90 | 74 / 65 | 138 / 116 |
| int4-g32-a16 with `--attention int` | 167 / 152 | 102 / 109 | 237 / 224 |

(3 interleaved rounds with Chrome and Emacs running, load 2–5.)

## Measuring accuracy

```sh
./gpt2t_bin --dtype FMT --ppl tests/data/alice_ch1.txt --compare
```

- **Evaluation text:** Chapter I of *Alice's Adventures in Wonderland* (public domain, Project Gutenberg eBook #11, header and license removed), 3,308 tokens.
- **Scoring:** sliding windows of 1,024 tokens every 512. Each token is scored once, with at least 512 tokens of context after the first window.
- **With `--compare`,** a float32 model runs on the same windows and the harness reports:

| Metric | Meaning |
|---|---|
| perplexity | exp(mean negative log-likelihood of the actual next token); lower is better |
| top-1 agreement | % of positions where the format and float32 predict the same most likely token |
| mean KL(f32 ‖ format) | how far the whole predicted distribution moves from float32's, in nats; the most sensitive measure |
| logit \|diff\| | mean and max absolute difference of the raw logits over the whole vocabulary |

## Verification

Build `gpt2t_bin` (and `gpt2_bin` for the first check) before running these:

| Check | Command |
|---|---|
| float32 through the generic code is bit-identical to the baseline | `tests/same_as_baseline.sh` |
| Logits, greedy decoding and perplexity match a NumPy GPT-2 | `uv run --with tiktoken python tests/reference.py {logits,greedy,ppl} ...` |
| Our quantization matches NumPy (add `--quant BITS,GROUP,SYM`, `--quant-head`, `--act16`, `--kv FMT`) | `uv run --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt --quant 4,32,0 --quant-head 8,0,0` |
| GGUF dequantization matches llama.cpp's Python `gguf` package | `uv run --with gguf python tests/gguf_check.py gpt2/gguf/gpt2.Q4_K_M.gguf` |
| GGUF perplexity matches NumPy (compare with `--kv f32`) | `uv run --with gguf --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt --gguf FILE` |
| Integer kernels match exact float64 math, including worst-case overflow inputs | `uv run mojo run -I . tests/a16_kernels.mojo` |
| Integer GGUF heads: decode and prompt paths identical, match exact math | `uv run mojo run -I . tests/gguf_int_head.mojo` |
| Integer softmax and integer attention match exact math | `uv run mojo run -I . tests/int_softmax.mojo`, `tests/int_attention_check.mojo` |
| Tokenizer matches `tiktoken` | `uv run --with tiktoken python tests/tokenizer_vs_tiktoken.py` |

Current results:
- float32 perplexity matches NumPy to 6 digits (25.284337 against 25.284352), and so do the quantized and GGUF formats.
- GGUF dequantization is within ~4×10⁻⁸ relative error.
- The integer kernels are within ~3×10⁻⁷ relative error.
- Greedy decoding matches NumPy token for token for every format checked.

[`PLAN.md`](PLAN.md#verifying-changes) has the exact commands and pass criteria.

## Performance notes

- **Set `MODULAR_THREAD_BUSY_WAIT_US=0`.** By default, idle Mojo worker threads spin between parallel regions. On a 15 W 4-core laptop the spinning threads take cycles and power from the working ones. This setting took float32 decode from ~38 to ~60 tok/s when decoding still used ~60 parallel regions per token. Decoding now uses one region per token, but prompts still use one per operation. The runtime reads it at startup, so it must be set in the environment. Machines with more cores may prefer a different value.
- **Decode threads:** `--threads N` sets the decode team size. The default, one per physical core, was ~40% faster than using every hyperthread on this CPU.
- **Benchmark on AC power and interleave runs.** On battery, long-context decode drops to about half. `tests/bench.sh` runs formats in alternating rounds, prints medians, and warns when on battery.
- **Decode is still below the memory limit** for the quantized formats: ~90 MB per token at ~260 tok/s is ~23 GB/s, against the ~44–55 GB/s the machine can stream.
- **The output head** (38.6M weights per token, 16-35% of decode) is the largest single operation. At int8 it runs within ~15% of the time a plain read of its bytes takes. GGUF heads (Q6_K, Q8_0) now run in integers: 2.1-2.8 ms down to 1.3-1.8 ms per token. They are still limited by arithmetic. (`research/test_head.mojo`)

## Project layout

| Path | What |
|---|---|
| `gpt2t.mojo` | The model (`Model[W, E, KV]`, including the team decode), loaders for safetensors and GGUF, sampling, evaluation, profiling, CLI |
| `tensor.mojo` | The `WeightMatrix` trait, `DenseMatrix`, `QuantMatrix`, `dot_pairs` (VNNI) |
| `gguf.mojo` | GGUF reader, `GGUFMatrix`, SIMD dequantizers for six block formats |
| `kernels.mojo` | matmul, GEMV, `matmul_rows`, `matmul_rows_a16`, register tiles, LayerNorm, attention, output head, and the one-token team kernels `linear_team` / `head_team` |
| `team.mojo` | `Team`: a spin-then-yield barrier for the threads that decode a token together |
| `kvcache.mojo` | The `KVCache` / `FloatKV` traits, the float attention, `DenseKV` and `QuantKV` |
| `int_attention.mojo` | `IntAttnKV`: int8 cache in VNNI layouts, integer attention |
| `intmath.mojo` | Integer `masked_exp` / `masked_softmax` in fixed point |
| `tokenizer.mojo` | GPT-2 byte-level BPE tokenizer |
| `gpt2.mojo` | The original single-file float32 program: the frozen reference |
| `tests/` | NumPy reference, tokenizer check, GGUF checks, kernel tests, baseline comparison, benchmark script, evaluation text |
| `research/` | Commented Mojo probes, each answering one question (see [`research/README.md`](research/README.md)) |
| `PLAN.md` | The shared plan: design decisions, milestone notes, full results, Mojo 1.1 notes, conventions |
| `pyproject.toml`, `uv.lock` | The pinned environment |

The weights (`gpt2/`), the environment (`.venv/`) and the build outputs are not in the repository.

Mojo 1.1 differs a lot from older Mojo, which most online examples use. `PLAN.md`'s [Mojo 1.1 notes](PLAN.md#mojo-11-notes) list the differences hit in this project: the `std.` import prefix, the new pointer API, closure capture lists, `comptime`, and others.

## Status and roadmap

| Milestone | State |
|---|---|
| M1. Generic weight formats, float32 bit-identical to the baseline | done |
| M2. Accuracy harness | done |
| M3. float16 / bfloat16, own int8 / int4 with int8 head, GGUF, fast kernels | done |
| M4. int16 activations with integer VNNI kernels (W4A16 / W8A16) | done |
| M5. Pluggable KV cache formats matched to the model's precision, and integer attention | done |
| M6. Tuning: profiling, one parallel region per decoded token, vectorized GELU, faster output heads, and a clean results re-run | done |
| M7. Stretch: save pre-quantized weights, int8 activations (VPDPBUSD) | planned |

Open questions (details in `PLAN.md`):
- A second evaluation text, to firm up the accuracy numbers.
- Why the importance-weighted GGUF file (i1-Q4_K_M) scores worse than plain Q4_K_M on this text.

## Credits

- **GPT-2 124M:** OpenAI's model and weights, via [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on Hugging Face.
- **GGUF files:** [mradermacher/gpt2-GGUF](https://huggingface.co/mradermacher/gpt2-GGUF), [mradermacher/gpt2-i1-GGUF](https://huggingface.co/mradermacher/gpt2-i1-GGUF) and [QuantFactory/gpt2-GGUF](https://huggingface.co/QuantFactory/gpt2-GGUF). The block formats are from [llama.cpp](https://github.com/ggml-org/llama.cpp)'s ggml.
- **Evaluation text:** *Alice's Adventures in Wonderland* by Lewis Carroll (public domain), from [Project Gutenberg](https://www.gutenberg.org/ebooks/11).
- **Built with** [Mojo](https://www.modular.com/mojo) 1.1 and Modular's MAX 26.6.
