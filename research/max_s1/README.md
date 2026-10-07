# S1: what MAX's Llama3 graph looks like (max 26.6.0)

`build_llama3_graph.py <out.mlir>` builds the legacy `max.pipelines.architectures.llama3.Llama3` graph
from a hand-made tiny config (2 layers, hidden 64, 4 heads, 2 KV heads, head_dim 16, vocab 256,
f32, `use_subgraphs=False`) with zero-filled weights, and prints the MO module. No HF download, no
compile, no device. Output: `llama3_tiny.mlir`.

Run it in a venv with `max==26.6.0 huggingface_hub transformers requests pillow av llguidance
pydantic ...` (importing `max.pipelines` pulls in the serving dependencies; `~/fun`'s env lacks them).

## Op table (whole 2-layer graph; weights are `mo.constant.external` by HF name)

| MO op | Count | What it is | hexagon_torch target |
|---|---|---|---|
| `mo.constant.external` | 21 | named weights (`layers.N.self_attn.q_proj.weight`, ...) | same names `lower.py` matches with `_LINEAR` |
| `rmo.mo.gather` | 2 | embedding lookup; last-token row select before the head | host; host/`hexagon` head slice |
| `mo.reduce.rms_norm` | 5 | RMSNorm, one op (gamma, eps) | `hexagon.rms_norm` (1:1) |
| `rmo.concat` of weights -> `transpose` -> `matmul` | q,k,v and gate,up are concatenated at graph level | fused QKV (`[128,64]`), fused gate/up (`[256,64]`) | `hexagon.linear`; the concat of constants must be traced back to the parts, or packed as one |
| `mo.custom mo.rope_split_store.ragged.paged` | 2 | splits fused QKV, applies RoPE to Q and K, stores K,V into the paged cache; returns rotated Q | `hexagon.rope` + `hexagon.index_copy` |
| `mo.custom mo.mha.ragged.paged` | 2 | causal attention over the paged cache | `hexagon.scaled_dot_product_attention` |
| `mo.split` + `rmo.mo.silu` + `rmo.mul` | 2 | SwiGLU on the fused gate/up output | `hexagon.swiglu` |
| `rmo.add` | 4 | residual adds | `hexagon.add` |
| `range/div/pow/cast/reshape/mul/cos/sin/concat` | one chain | builds the `[max_seq, head_dim]` interleaved cos/sin table (`freqs_cis`) from constants | constant-fold on the host into our cos/sin buffers |
| `rmo.slice`, `rmo.sub` | 1 each | `input_row_offsets[1:] - 1`: the last-token index | host |

About 14 distinct ops in all; only the two `mo.custom` ops are not plain tensor ops.

## Graph inputs (what the "ragged + paged" interface costs us)

```
tokens                [total_seq_len]            i64   all sequences concatenated
input_row_offsets     [n_seqs+1]                 u32   sequence boundaries
return_n_logits       [n]                        i64
kv blocks (buffer)    [pages, 2, layers, 128, n_kv, head_dim]   page_size 128
lookup_table          [batch, max_pages]         u32   page table
cache_lengths         [batch]                    u32
max_lengths           [1,1]
rope scale            [n_heads]                  i64
```

Static single-sequence mapping: batch = 1, `input_row_offsets = [0, t]`, an identity page table,
and our own packed KV in place of the paged buffer. `cache_lengths` carries `pos`.

## Findings against the S1 gate

- **Op coverage is small and clean.** RMSNorm is 1 op (vs 6 aten ops that `lower.py` matches);
  SwiGLU and residual adds are plain; RoPE + KV store and attention are single named custom ops.
  `lower.py`'s RMSNorm, RoPE (chunk/mul/sub/cat) and KV/SDPA regex-style patterns become a lookup
  by op name.
- **Paged KV is tractable.** The custom ops take the cache as an opaque buffer plus page table;
  we do not need to emulate paging, only map the two ops onto our static cache with batch 1.
- **Not free.** (1) QKV and gate/up are concatenated *in the graph*, so a converter must follow
  `concat` of external constants (cheap: names are intact) and our fused/split linears must match
  MAX's output column order (q|k|v, gate|up). (2) RoPE is half-split vs interleaved: the op
  carries `interleaved=false`, our `hexagon.rope` is half-split; other models set it to true.
  (3) The cos/sin table is computed in-graph from `range/pow/cos/sin` and has to be constant
  folded (numpy evaluation of ~10 ops). (4) The default config uses `use_subgraphs=True`
  (`mo.call` into layer subgraphs); here it is off. (5) f32 here; a bf16 model adds `mo.cast`.
- **Dependency footprint is heavy.** `import max.pipelines` needs pydantic, pillow, av,
  llguidance, transformers, huggingface_hub, requests, ... even for graph construction. Building
  from `max.nn` directly avoids some of it but not the paged-KV types.

## Not done in S1
- The ModuleV3 variant (`llama3_modulev3`); Qwen3 (QK-norm: `fused_qk_rms_norm_rope_ragged`?).
- Real weights through MAX's weight adapters (this graph used zeros).
- Numerics: no execution, nothing compared.

---

# S1b: Qwen3 (`build_qwen3_graph.py`, `qwen3_tiny.mlir`, `op_table.py <mlir> [1]`)

Same tiny config (head_dim 16, 4 heads, 2 KV heads), tied embeddings, rope theta 1e6. The legacy
`Qwen3` class is GPU-first: its `Allreduce` layer opens an `Accelerator` even on one device, so the
script patches `Allreduce.__init__` (single device never calls it). The model also takes a
`signal_buffers` graph input (here one unused buffer). `op_table.py` prints the op histogram.

## Differences from Llama3 (the whole graph is otherwise the same op set)

| Qwen3 detail | In the graph | Converter consequence |
|---|---|---|
| QK-norm | `mo.reduce.rms_norm` on `[t, heads, head_dim]` for Q (4 heads) and for K (2 heads), gamma `[head_dim]`, **before** RoPE; results reshaped and `concat`ed with V into `[t, 128]` | maps to `hexagon.head_rms_norm` (already in `lower.py`); the concat feeds the RoPE op |
| RoPE + KV store | the same `mo.rope_split_store.ragged.paged` (`interleaved=false`) takes the *post-norm* concatenated QKV | split the concat back into q/k/v, then `hexagon.rope` + `hexagon.index_copy`; the QK-norm must be emitted first |
| QKV split | `mo.split` into 3 (`[64, 32, 32]`) | follow the three weight slices, as for Llama3 |
| Weight access | every weight goes through `rmo.slice` (the vocab/column-parallel shard of a 1-device layout: a full slice) | look through no-op slices of `mo.constant.external` |
| Embedding | masked gather: `slice`, `greater_equal` x2, `not`, `and`, `sub`, `mul`, `gather`, `cast`, `mul` (the vocab-shard mask, all-true on one device) | host op; recognise the pattern or treat as plain `embedding` |
| LM head | tied: `transpose(embed_tokens)` then `matmul`; no `lm_head.weight` | head linear from the embedding weight, as `hexagon_torch` does for tied models |
| Final head | `gather(last token)`, `rms_norm`, matmul (same as Llama3) | unchanged |

Op counts for the whole 2-layer graph: 9 `rms_norm` (4 block norms + 4 QK-norms + final), 9 `matmul`,
4 `split`, 2 each of `rope_split_store` and `mha`, 2 `silu`, 4 `add`, plus ~26 `slice` and the cos/sin
table chain. No op appears that Llama3 did not already have, apart from the vocab-mask ops
(`greater_equal`, `not`, `and`).

## Verdict

Qwen3 adds nothing structurally new to the converter: QK-norm is an existing `rms_norm` op on a
3-D view, and RoPE/attention are the same two custom ops. Both models reduce to ~15 op kinds.
The real costs are in tooling: the legacy Qwen3 class needs a GPU stub to build on CPU, and the
sharded-layer wrappers (slices, masked gather) add noise that the converter has to see through.
The ModuleV3 variants (`qwen3_modulev3`) may be cleaner and were not looked at.
