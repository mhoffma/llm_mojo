# MAX front end for hexagon_torch

Build the model graph with MAX (`max.nn` / `max.pipelines`), convert it to `hexagon::` ops, and hand
that program to the existing hexagon_torch back end (memory plan, `.hxb`, device server, DSP kernels).
It replaces the `torch.export` front end and the pattern matching in `hexagon_torch/lower.py`.
Design question and trade-offs: [`../../max_hexagon_scope.md`](../../max_hexagon_scope.md).

![Path from a Hugging Face checkpoint to the Hexagon cDSP: the MAX front end and the torch.export front end meet at one hexagon:: program, then hexagon_torch builds an .hxb, the device server loads it, and the Mojo kernel skeleton runs it on the cDSP.](max_path.svg)

*Teal: new code. Dashed: not yet run from the MAX front end. The picture's source is
[`max_path.svg`](max_path.svg); [`path_diagram.html`](path_diagram.html) is the same picture as a
page, generated from it.*

## Status (2026-10-07)

| Step | What | State |
|---|---|---|
| S1 | MAX's Llama3 and Qwen3 graphs read into an op table ([`../max_s1`](../max_s1/README.md)) | done |
| S2 | `max_lower.py`: a Llama3 decoder block equals `lower.py`'s output | done, host emulator |
| S3a | TinyLlama, 22 layers, 12 greedy tokens equal to the fp32 PyTorch reference | done, host emulator |
| S3b | static-shape export, memory plan, `.hxb`, run on the DSP | **not started** (needs a `Session` on the shared board `vq`) |
| later | Qwen3 (per-head RMSNorm, sliced weights), W4 weights, ModuleV3 models | open |

## Use

```sh
# venv: max==26.6.0, CPU torch, pytest, numpy, huggingface_hub, transformers, requests, pillow, av,
# llguidance, pydantic (importing max.pipelines pulls in the serving dependencies)
export HVXHMX_REPO=<checkout of mhoffma/hvxhmx_mojo>          # supplies hexagon_torch

python -m pytest -q -s test_max_lower_block.py -p no:logging  # one block vs lower.py and PyTorch
python run_tinyllama.py <TinyLlama-1.1B-Chat-v1.0 dir> "Qualcomm is" 12 22 out.json   # ~1 min pack, ~30 s/token
python ref_tinyllama.py <same dir> "Qualcomm is" 12 ref.json                          # fp32 reference
```

| File | Role |
|---|---|
| `max_lower.py` | the converter: MO ops of a `max.graph.Graph` -> a torch.fx program of `hexagon::` ops |
| `build_block.py`, `build_model.py` | MAX graphs: one decoder block with given weights; a whole model from a HF `config.json` with lazy safetensors weights |
| `test_max_lower_block.py` | the S2 gate |
| `run_tinyllama.py`, `ref_tinyllama.py` | the S3a run and its reference; logs in `results/` |
| `max_path.svg`, `build_diagram.py`, `path_diagram.template.html`, `path_diagram.html` | the picture above and its page |

## Maintaining this document and the picture

- **The picture is part of the feature.** When a stage changes state (S3b runs, a dashed box becomes
  solid), a stage is added or renamed, or the flow changes, edit `max_path.svg` in the same commit as
  the code or the status table above. Keep the picture and the table saying the same thing.
- Edit only `max_path.svg`; then `python3 build_diagram.py` regenerates `path_diagram.html`. Never
  edit the generated page by hand.
- The published page (a private Claude artifact, `https://claude.ai/artifact/Edz4sarvN8sQz4C6nuGixS`)
  is republished from `path_diagram.html` after a regeneration.
- The diagram's colours are theme tokens at the top of the SVG's `<style>`; light and dark are both
  defined there. The standalone file was checked in light mode in headless Chrome; dark mode was not
  confirmed.
- Numbers quoted in the picture (12/12 tokens) come from `results/`; update them with the result.

---

# S2: a MAX Llama3 decoder block, lowered onto `hexagon::` ops

`max_lower.py` walks the MO ops of a `max.graph.Graph` and emits a torch.fx program of `hexagon::` ops
(hexagon_torch's own op library, emulator backend). `build_block.py` builds the MAX graph of one
`max.nn` Llama3 TransformerBlock with given weights. `test_max_lower_block.py` is the S2 gate.

```sh
# a venv with: max==26.6.0 torch (cpu) pytest numpy huggingface_hub transformers requests pillow av llguidance pydantic ...
# (see ../max_s1/README.md for the import dependencies)
HVXHMX_REPO=<checkout of mhoffma/hvxhmx_mojo> python -m pytest -q -s test_max_lower_block.py -p no:logging
```

## Result (2026-10-07, x86, emulator; same block as `tests/test_lower_block.py`: hidden 512, 8/4 heads, ffn 1408, t=64)

| Check | Result |
|---|---|
| op multiset vs `lower.lower()` of the torch block | **identical**: 7 linear, 2 rms_norm, 1 swiglu, 2 rope, 2 index_copy, 1 sdpa, 2 add |
| output vs `lower.py`'s program | max abs diff 1.95e-3, 92.0% bit-equal; **100.0% bit-equal when MAX's cos/sin table is replaced by the reference table** |
| output vs the PyTorch reference block | within the same 2e-2 tolerance `test_lower_block.py` uses |
| packed K/V caches vs the reference caches | within 2e-2; positions >= t untouched (zero) |

The 8% is the table: MAX computes `freqs_cis` in fp32 in the graph (`range * inv_freq` in fp32, then
cos/sin), the reference in fp64 then rounds; the tables differ by up to 9e-6, which flips a few
fp16 last bits. Nothing else differs; the converter's DSP op sequence equals `lower.py`'s.

## What the converter does (and what it needed)

| MO | Emitted |
|---|---|
| `mo.constant`, `mo.constant.external`, and any op on constants only (`range div pow mul add sub cast cos sin reshape concat transpose`) | numpy-folded. This evaluates MAX's cos/sin table chain and the q\|k\|v and gate\|up weight concatenations. |
| `mo.reduce.rms_norm` (2-D) | `hexagon.rms_norm` |
| `rmo.matmul` by a constant | a deferred linear: emitted as one `hexagon.linear` when consumed |
| `mo.split` of a matmul (gate\|up) | **one `hexagon.linear` per piece** (weight columns sliced), no split on the DSP |
| `mo.custom rope_split_store.ragged.paged` on the fused QKV matmul | q, k, v linears (from the head counts in the KV buffer type), `hexagon.rope` on q and k, `hexagon.index_copy` for k and v. Table = `freqs[:max_seq]`, cos at even columns, sin at odd. |
| `mo.custom mha.ragged.paged` | `hexagon.scaled_dot_product_attention` (causal, scale from operand 8) |
| `silu` then `mul` | `hexagon.swiglu` |
| `rmo.add` | `hexagon.add` |
| `rmo.reshape` | alias (the DSP values are `[t, features]`) |

Anything else raises `NotImplementedError` naming the MO op, so unsupported models fail loudly.

## Specialization and limits

- `t`, `pos` and `max_seq` are converter arguments; MAX keeps them symbolic (`total_seq_len`,
  `cache_lengths`). Batch 1, one prefill block. The packed caches are new zero buffers, not MAX's
  paged buffer.
- MAX's rope module builds the table for **twice** the context (512 rows for `max_seq` 256): sliced.
- fp32 graph, fp16 weights/activations on the DSP (as `lower.py`'s default `f16`). No W4, no bias, no
  `residual_multiplier`/`embedding_multiplier` handling yet.
- Per-head `rms_norm` (Qwen3 QK-norm) and `rmo.slice` of weights are not handled: S2 is Llama3 only.
- Runs on the host emulator only; the DSP path is the same ops as `lower.py`'s, not retested here.
- Needs the graph's live MLIR (`Graph._body`), a private API of max 26.6.0.

---

# S3a: TinyLlama decode through the MAX front end (host emulator)

`build_model.py` (whole `max.pipelines` Llama3 graph from a HF `config.json`, weights served lazily
from safetensors), `run_tinyllama.py` (greedy decode), `ref_tinyllama.py` (the fp32 PyTorch
reference, hexagon_torch's `models/llama.py`). Logs and token lists: `results/`.

```sh
HVXHMX_REPO=<repo> python run_tinyllama.py <TinyLlama-1.1B-Chat-v1.0 dir> "Qualcomm is" 12 22 out.json
HVXHMX_REPO=<repo> python ref_tinyllama.py <dir> "Qualcomm is" 12 ref.json
```

## Result (2026-10-07, x86 host emulator, 22 layers, fp16 weights, 12 greedy tokens)

| | |
|---|---|
| Graph → program | 22-layer MAX graph (153 KB of MLIR) → one program: 155 linear, 45 rms_norm, 44 rope, 44 index_copy, 22 sdpa, 22 swiglu, 44 add (the converter reports exactly the ops of `lower.py` for 22 layers + head) |
| Generated text | `Qualcomm is a leading provider of wireless technology and services, and is a` |
| vs fp32 PyTorch reference | **12 / 12 tokens identical**; top-5 identical at every step |
| Logits | max abs diff 0.010-0.026 (logit range 17.2: 0.15%); cosine similarity >= 0.999998; the reference's smallest top-1/top-2 margin is 0.069, so the equality has headroom |
| Cost | convert + pack (1.1B weights → HMX fp16): 56 s; 27-48 s per token on this machine while another job was running (the host emulator, not the DSP) |

What the full model needed beyond S2 (all in `max_lower.py`): runtime `pos` and row count (one program
serves the prompt call and every decode step), per-layer KV buffers (the layer index is a constant
operand of the custom ops), the embedding as a host gather, the last-token row select
(`input_row_offsets[1:] - 1` recognised and turned into a slice), the head, lazy weights that are freed
after their last use (peak memory stays at one layer's fp32 weights plus the packed fp16 buffers), and
a fix to the type parser for dims like `add(n, -1)`.

## What this does not show yet (S3b)

- **Not the DSP and not an `.hxb`.** The stated S3 gate is "tokens identical to the current `.hxb`".
  This is the host half: the converted program's results match the PyTorch reference through the same
  `hexagon::` emulator the DSP is checked against. The next step is to turn the program into an
  `ExportedProgram` with static shapes (prefill rows and 1 decode row), plan it with `memplan`, build
  `Program`s, and export an `.hxb` with `export_blob` — which needs a `Session` on `vq` (shared board:
  own port, own directories; `BUILD.md` section 6).
- **Not compared against `lower.py`'s program on the full model**, only on one block (S2: bit-equal
  with the same RoPE table). A fp32-reference comparison is weaker than a bit comparison.
- **The gate is greedy equality over 12 tokens on one prompt**, with a 0.069 minimum margin.
