# MAX front end for hexagon_torch

Build the model graph with MAX (`max.nn` / `max.pipelines`), convert it to `hexagon::` ops, and hand
that program to the existing hexagon_torch back end (memory plan, `.hxb`, device server, DSP kernels).
It replaces the `torch.export` front end and the pattern matching in `hexagon_torch/lower.py`.
Design question and trade-offs: [`SCOPE.md`](SCOPE.md).

![Path from a Hugging Face checkpoint to the Hexagon cDSP: the MAX front end and the torch.export front end meet at one hexagon:: program, then hexagon_torch builds an .hxb, the device server loads it, and the Mojo kernel skeleton runs it on the cDSP.](max_path.svg)

*Teal: new code. Every box has run from the MAX front end. The picture's source is
[`max_path.svg`](max_path.svg); [`path_diagram.html`](path_diagram.html) is the same picture as a
page, generated from it.*

## Status (2026-10-07)

| Step | What | State |
|---|---|---|
| S1 | MAX's Llama3 and Qwen3 graphs read into an op table ([`probe_graphs/`](probe_graphs/README.md)) | done |
| S2 | `max_lower.py`: a Llama3 decoder block equals `lower.py`'s output | done, host emulator |
| S3a | TinyLlama, 22 layers, 12 greedy tokens equal to the fp32 PyTorch reference | done, host emulator |
| S3b | static-shape export, memory plan, `.hxb`, run on the DSP | done, fp16 and W4: 12/12 tokens on the DSP; same packed weights as the stock containers |
| Qwen3 | Qwen3-0.6B: per-head QK-norm, sliced head, tied embeddings, fp16 and the W4 recipe | done: 12/12 tokens on the DSP |
| later | int8 KV caches, ModuleV3 models, longer prompts and a quality measure (perplexity) for the MAX-made W4 containers | open |

Last verified 2026-10-07 against `mhoffma/hvxhmx_mojo` `037b8f0` (and a DSP library built from it): `test_max_lower_block.py` 3 passed
(max |diff| vs `lower.py` 1.95e-3, 92.0% bit-equal; the earlier analysis was against `1706124`).

## Use

```sh
# venv: max==26.6.0, CPU torch, pytest, numpy, huggingface_hub, transformers, requests, pillow, av,
# llguidance, pydantic (importing max.pipelines pulls in the serving dependencies)
export HVXHMX_REPO=<current checkout of mhoffma/hvxhmx_mojo>   # supplies hexagon_torch (it is not in this repo)

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
| `probe_graphs/` | S1: the tiny Llama3 and Qwen3 MAX graphs as MLIR, their build scripts and an op-table printer |
| `SCOPE.md` | the scoping paper: options, trade-offs, step plan |
| `gen_max.py` | `MaxGenerator`: hexagon_torch's `Generator` with its layer-group and head programs made from MAX graphs |
| `export_mojomax.py` | exports an `.hxb` with `MaxGenerator` (or the stock front end with `--stock`) through a device server |
| `run_dsp.py` | runs an `.hxb` on the DSP through a device server and compares the greedy tokens with `results/s3_ref.json` |
| `test_static_program.py` | S3b offline gate: the MAX program equals `lower.lower()`'s for a group of layers and for the head |
| `compare_hxb.py` | compares two `.hxb` containers section by section |
| `max_path.svg`, `build_diagram.py`, `path_diagram.template.html`, `path_diagram.html` | the picture above and its page |

## Maintaining this document and the picture

- **The picture is part of the feature.** When a stage changes state (a new stage is added dashed until it
  has run, then drawn solid), a stage is renamed, or the flow changes, edit `max_path.svg` in the same commit as
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
# (see probe_graphs/README.md for the import dependencies)
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

---

# S3b: the MAX programs on the DSP (TinyLlama, fp16 and W4 weights)

`gen_max.MaxGenerator` overrides `Generator._program`, which turns a module (layers [i0, i1) or the head)
into a lowered, planned program for one row count. It builds a MAX graph of the same layers
(`build_model.build_layers` / `build_head`), converts it with `max_lower.convert(..., batched=True)`, exports
it and plans it with `memplan`. Everything after that is hexagon_torch's own: `Program`, the shared KV state,
six 4-layer prefill programs of 32 rows, the 22-layer decode program, the head, `export_blob`.

## Results (2026-10-07, vq: QCS8300, Hexagon v75, 1.4976 GHz; DSP library built from `037b8f0`, sha256 `b4b08e78...`)

| Check | Result |
|---|---|
| Offline gate (`test_static_program.py`, 2 layers at 32 rows and at 1 row, and the head) | the 34 hexagon ops per program have identical arguments; every constant tensor (packed weights, gammas, bias tables, RoPE tables, initial caches) is identical byte for byte; the VTCM plan is identical (1,593,344 bytes peak). Two documented differences: MAX's own RoPE tables (below) and one shared cos/sin pair per layer instead of one per RoPE call |
| Export through a device server on vq | 22 layers, 155 linears; DSP held 164 s; container 2,104.98 MB, 13 sections, the stock export's size |
| Container vs the stock export (`results/s3b_container_compare.txt`) | 12 of 13 sections byte-identical: metadata, head weights, embedding, decode, head and prefill schedules, placement and patch tables. The weights arena differs in 19,865 of 1.94 GB bytes, all inside one 0.13 MB span: the RoPE tables (MAX builds them in fp32, the reference in fp64; max difference 1.8e-5). `exact_tables=True` should make them identical; not re-exported to confirm |
| Tokens on the DSP, prompt "Qualcomm is", 12 greedy | MAX container 12/12 identical to the fp32 PyTorch reference; stock container 12/12 identical too |
| Speed, 4-token prompt | MAX: TTFT 99-104 ms, decode 87.8-92.7 ms/token (4 runs). Stock: TTFT 98-103 ms, decode 87.6-88.8 ms/token (4 runs). The same schedules: the differences are the shared board's load |

### W4 (Q4_0 weights expanded to fp16 on the DSP; `--weights w4f16v2 --head-weights w4f16v2`)

The converter takes each linear's format from hexagon_torch's `WeightPolicy`, looked up by the name `lower.py`
would see (`blks.L.attn.q_proj.weight` ...). It knows the role of each linear from the graph (q, k, v from the
fused QKV that `rope_split_store` splits; gate, up from the `mo.split`; o after attention; down after SwiGLU;
the head otherwise), so keep-fp16 policies work too.

| Check | Result |
|---|---|
| Offline gate, 2 layers, 32 rows and 1 row, policies `f16`, `w4f16v2` and a mixed one (keep `k` and `L1.down` fp16), and the head in `f16` and `w4f16v2` | 8 tests pass: ops, constants (the Q4_0 packed bytes included) and VTCM plans identical; W4 peak 1,589,248 bytes |
| Containers (`results/s3b_w4_container_compare.txt`) | 748.54 MB each (fp16: 2,104.98 MB); 12 of 13 sections byte-identical, the weights arena (606 MB) differs only in the same 19,865 bytes of RoPE tables |
| Export | packing 156-198 s on 6 processes; DSP held 66 s (MAX) and 50 s (stock) |
| Tokens on the DSP, prompt "Qualcomm is", 12 greedy | MAX W4 = stock W4 = the fp32 reference, 12/12 |
| Speed, 4-token prompt | MAX: TTFT 55-59 ms, decode 46.9-47.4 ms/token (21.1-21.3 tok/s, 3 warm runs; a first cold run 52.0 ms). Stock: TTFT 56-58 ms, decode 47.1-47.6 ms/token. The same speed; 1.9x faster than fp16 |

## Run it

```sh
export HVXHMX_REPO=<current checkout of mhoffma/hvxhmx_mojo>
make -C $HVXHMX_REPO/hmx/layer liblayer_skel.so          # the DSP library; note its sha256
# vq: your own directory, the library there and in the content store /tmp/vq_skels/<sha256>/, your own server
#   (never the shared one on 9870): copy device_control/, write backend/active_backend.mojo as in BUILD.md
#   section 6.2, build it, then START IT AND KEEP ITS PID:
#       VQ_SERVER_PORT=9872 HEXAGON_LIB_DIR=... ./server_bin > server.log 2>&1 & echo $! > server.pid
python export_mojomax.py --model <TinyLlama dir> --weights w4f16v2 --head-weights w4f16v2 --out x.hxb --remote tcp://10.168.168.32:9872 \
    --lib-dir /home/mhoffman/hexagon/mojomax/lib --skel-hash <sha256>        # waits while another user holds the DSP
scp x.hxb vq:hexagon/mojomax/exports/
python run_dsp.py --container /home/mhoffman/hexagon/mojomax/exports/x.hxb --skel <sha256> --port 9872 --model <dir>
kill $(cat server.pid)                                    # on vq, when done: by PID, never by name
```

**Never stop a server by name or pattern on vq.** On 2026-10-07 a `pkill -f "^./server_bin"` meant for the test
server above also stopped the shared server on 9870 (and the sessions it had forked) for about two minutes;
it was restarted with `cd ~/hexagon/device_control && nohup ./server_bin >> server.log 2>&1 &` (the Makefile's
`start-server`), and a copy of its log, `server.log.before-restart-1791373867`, was left next to it. Record the
PID when you start a process and kill that PID.

## Limits

- fp16 and Q4_0 weights (fp16 or W4 head, a keep-fp16 policy); no int8 KV; short prompts (3-4 tokens) and 12 generated
  tokens per check; Llama-family and Qwen3 graphs from MAX's legacy models only.
- The test runs against a DSP library built from the same `hvxhmx_mojo` commit as the Python side; a library
  built from a different commit may not match the container's patch tables.

---

# Qwen3 (0.6B) through the MAX front end

MAX's legacy `Qwen3` model builds a graph that is Llama's plus per-head QK-norm, an explicit `head_dim` (128
here, wider than hidden / heads), tied embeddings and a vocabulary of 151,936. `build_model` builds its
layer-group and head graphs (it replaces the model's `Allreduce` layer, which opens a GPU even on one device
and is never called there). `max_lower` gained what Qwen3 needs: constant slices (the one-device shards of
every weight are full-range slices), `mo.reduce.rms_norm` on `[t, heads, head_dim]` -> `hexagon.head_rms_norm`,
a 3-way QKV split whose q and k are normed and re-concatenated before `rope_split_store` (the v linear is
emitted on first use, so the op order is `lower.py`'s), the layer number from a counter (Qwen3 splits QKV
before the RoPE op sees the layer index), and a head wider than one DSP tensor emitted as slices of 32,768
joined by a `cat`, as `models.llama.Head` does.

## Results (2026-10-07, vq; checkpoint Qwen/Qwen3-0.6B, 28 layers, context 512, prefill 32 rows)

| Check | Result |
|---|---|
| Offline gate (`test_static_program.py` with `MODEL_DIR=<Qwen3-0.6B>`: 2 layers at 32 and 1 rows in fp16, W4 and the W4 recipe; the head in fp16 and W4, 5 vocabulary slices) | 8 pass: 38 hexagon ops per layer program identical to `lower.py`'s, constants identical, VTCM plans identical |
| fp16 containers (`results/q3_f16_container_compare.txt`) | 1,440.12 MB (MAX) and 1,440.24 MB (stock). Head weights, embedding, head schedule, placement and patch tables are byte-identical. The weights arena differs by 131,072 bytes and the decode and prefill schedules by the table offsets that follow, **because of a quirk of the stock export**: its layer 0 uses a RoPE cos table that is one fp32 ULP off in 421 of 32,768 entries, layers 1-27 the exact one, so the stock arena holds two cos tables where MAX holds one (MAX's own fp32 table, for every layer) |
| Tokens on the DSP, prompt "Qualcomm is", 12 greedy | MAX 12/12 and stock 12/12, both equal to the fp32 PyTorch reference: "Qualcomm is a company that provides software and hardware solutions for the education sector" |
| Speed, 3-token prompt (3 alternating runs each) | MAX: TTFT 66.5-69.2 ms, decode 59.7-61.9 ms/token (16.2-16.7 tok/s). Stock: TTFT 64.8-75.6 ms, decode 58.2-61.7 ms/token. Ranges overlap |

### Qwen3 W4 recipe (`--weights w4f16v2 --head-weights w4f16v2 --keep-f16 k,v,L2.down`)

| Check | Result |
|---|---|
| Containers | 739.73 MB (MAX) and 739.85 MB (stock) |
| MAX container with MAX's own fp32 RoPE tables | **diverged from the stock W4 container and the fp32 reference at token 10** ("...solutions for educational institutions." against "...for the education sector"): 9/12 |
| Cause | Page by page (4 KiB, by content hash) the two weights arenas hold the same packed weights; they differ only in the RoPE tables (MAX's fp32 table is up to 2.8e-5 off the reference's). Re-exporting the MAX container with the reference's tables (the default now, `exact_tables=True`; `--max-tables` keeps MAX's) gave 12/12. At fp16 the same table difference changes no token; Q4_0 flips a near-tie |
| MAX container, reference tables | 12/12 identical to the stock W4 container and to the fp32 reference: "Qualcomm is a company that provides software and hardware solutions for the education sector". The MAX arena has **no page that the stock arena lacks**; the stock arena has 32 more (its layer-0 cos table, see above) |
| Speed, 3-token prompt (3 alternating runs each) | MAX: TTFT 46-54 ms, decode 40.7-42.5 ms/token (23.5-24.6 tok/s). Stock: TTFT 44-48 ms, decode 40.4-41.3 ms/token. fp16 for comparison: about 60 ms |

The table episode is worth remembering: a numerically small difference in a constant can move a greedy token
when the weights are quantized. The MAX path therefore defaults to the reference's tables so its containers can
be compared with the stock ones token for token. Whether MAX's own table is better or worse for quality is not
measured (no perplexity run); the two outputs at token 10 are equally plausible.
