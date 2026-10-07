# Scoping: MAX as an alternative path onto the Hexagon cDSP

Status: scoping draft, 2026-10-07. Based on a read of `mhoffma/hvxhmx_mojo` (`hexagon_torch/`,
`device_control/`, `BUILD.md`) and the installed `max 26.6.0` / `mojo 1.1.0` in `~/fun/.venv`.
The first sections were written before anything ran on the board; steps S3a and S3b of the plan at the end later did (on 2026-10-07). Items marked **[verify]** are unconfirmed.

![The path from a checkpoint to the Hexagon cDSP with MAX as the front end](max_path.svg)

*The two front ends meet at one `hexagon::` program. Teal is new code; every box has run from the
MAX front end (host emulator and DSP). Source: `mojo.max/max_path.svg`; the implementation notes are in
`mojo.max/README.md`.*

## 1. Question

`hexagon_torch` lowers `torch.export` graphs onto fused HMX/HVX operators on the Hexagon v75 cDSP.
Could the MAX graph stack (`max.graph`, `max.nn`, the `max.pipelines` model zoo) be an alternative
front end, runtime, or both? What would it give us, and what would it cost?

## 2. Where hexagon_torch's value sits

The pipeline is layered. Only the top layer is the part MAX could replace.

| Layer | Files (lines) | Replaceable by MAX? |
|---|---|---|
| Model capture: `torch.export` of Llama/Qwen3 reference modules | `reference/`, `models/` | **Yes**: this is MAX's job (`max.nn`, `max.pipelines.architectures`) |
| Lowering: aten graph -> `hexagon::` ops by pattern matching (RMSNorm, SwiGLU, RoPE, KV cache + SDPA, residual add, FFN split) | `lower.py` (564) | **Yes**, in principle: the same job on a different IR |
| Weight packing: Q4_0 / fp16 into HMX layouts | `pack.py`, `layouts.py`, `q4_0.py` | No. Hardware-specific. |
| Memory plan for 8 MB VTCM, schedule bytecode, `.hxb` container | `memplan.py`, `schedule.py`, `export_blob.py` | No. MAX's planner knows nothing about VTCM. |
| DSP kernels and executor (HMX streaming matmul, HVX ops, `hwsched`) | `hmx/layer/*.mojo` | No. |
| Device server, FastRPC, DMA arenas | `device_control/` | No (but see option B). |

So MAX can replace roughly 1,000 lines of Python at the top, not the stack. The case for it has to
rest on what those lines cost us today: `lower.py` is regex and pattern matching over aten graphs,
tuned to two model families, and every new architecture risks a new pattern.

## 3. What MAX provides (observed in 26.6.0)

- **A Python graph builder** (`max.graph`): `Graph`, `TensorType` with symbolic dims, ~70 op
  modules (`rms_norm`, `matmul`, `silu`, `gather`, `concat`, ...). Building a graph needs no
  device and no compile. I built an RMSNorm -> matmul -> SiLU graph on this x86 box and printed it:
  ```
  mo.graph @blk<t>(%arg0: !mo.tensor<[t, 64], f32>) -> !mo.tensor<[t, 128], f32>
    %5 = mo.reduce.rms_norm(%arg0, %2, %3, %4) {multiply_before_cast = false}
    %6 = rmo.matmul(%5, %1)
    %7 = rmo.mo.silu(%6)
  ```
  `Graph.module` exposes the MLIR module (`_to_mlir_str()`, and live ops through `max.mlir`). RMSNorm
  and SiLU arrive as single named ops, with no aten decomposition to pattern-match back together.
  Dims such as `t` stay symbolic, which fits our prefill/decode split.
- **A model zoo** (`max.pipelines.architectures`): llama3, qwen3-class, gemma3/4, mistral, granite,
  olmo, and more, in both a legacy and a "modulev3" form, with weight adapters for safetensors and
  GGUF. This is the largest potential gain: model coverage we would not write ourselves.
- **Quantization hooks**: `ops.quantized` repacks GGUF weights (Q4_0 and others), and `nn` has
  quantized linears. Our W4 path already starts from Q4_0, so GGUF-direct export looks natural.
- **Custom ops**: `ops.custom(name, device, values, out_types, parameters)` inserts an op backed by a
  Mojo kernel registered with `@compiler.register`, plus `inplace_custom` for buffer updates.
- **Torch interop**: `max.experimental.torch.graph_op` wraps a MAX graph as a torch custom op.
  That is the reverse direction (MAX inside torch), not useful here.
- **Cross-host CPU codegen**: `driver.set_virtual_cpu_target("neoverse-n1" | "generic" | ...)`
  compiles CPU kernels for a fixed target instead of the build host. This could help the CPU-side
  work of a MAX model run on the aarch64 board while building on x86.

## 4. What MAX does not provide

1. **No Hexagon backend.** `mojo build --target-triple hexagon-unknown-none-elf` fails with
   "target ... is not supported by this build". Devices are CPU and GPU only. `libmax.so` contains
   Hexagon LLVM strings, but I could not turn that into a usable target **[verify with Modular]**.
   The riscv32-IR retarget trick (`BUILD.md` section 3) therefore stays our DSP code path under every
   option below.
2. **The model zoo is built for GPU-style serving.** The fused ops are `fused_qkv_ragged_matmul`,
   `rope_ragged`, `kv_cache_store_paged_ragged` and flash attention over paged KV collections. They
   assume ragged batches and a paged KV manager. Our DSP wants one static, HMX-layout KV cache per
   layer, single stream. Mapping paged-KV ops onto that is the main technical risk (section 6).
   The padded variants (`fused_qkv_padded_matmul`, `kv_cache_store_paged_padded`,
   `flash_attention_padded_kv_cache`) are closer, but still paged.
3. **Stability.** The MO dialect and `max.mlir` are not a documented public interface. This is
   26.6 and the API moves quarter to quarter. We would pin the version, as `~/fun` already does.
4. **Platform.** The wheel here is `manylinux_2_34_x86_64`. Whether MAX installs on the board
   (aarch64 Ubuntu) is **[verify]**. `BUILD.md` says Mojo already runs on `vq`, so Mojo alone is fine.

## 5. Options

### A. MAX as the front end only (recommended spike)
Build or load the model with `max.nn` / a pipelines architecture, take the `Graph`, and write a
MO -> `hexagon::` converter that feeds the existing `memplan` / `schedule` / `export_blob`. Run on
the x86 build machine, offline, as today.

- Keeps everything below the lowering untouched.
- The converter is a table over named MO ops, not regexes over decomposed aten.
- Needs a static-cache attention form. Either write our own small `max.nn` decoder with a plain
  static KV (cheapest, loses part of the zoo), or implement converters for the paged/ragged custom
  ops and treat them as a static cache with one sequence.
- Weights: load through MAX's weight adapters, then pack with our existing packers.

### B. MAX as the host runtime, DSP as a coarse custom op
Run MAX on the board and register one Mojo custom op, "run this `.hxb` schedule", so the DSP model
is a node in a MAX graph. CPU parts (embedding, sampling) and serving (`max.serve`) come from MAX.

- Must stay coarse: the DSP design depends on one schedule per token or chunk, with
  sub-millisecond dispatch. A per-op FastRPC call would be far slower.
- MAX's memory planner cannot see VTCM or the rpcmem weight arena, so those stay ours.
- Depends on MAX running on aarch64 **[verify]**, and the device lock in `device_control` has to be
  respected.
- Worth it only if we want MAX's serving and OpenAI-style API. `device_control` already does
  serving.

### C. Compile DSP kernels from MAX's own kernel library
Not recommended. MAX's generic Mojo kernels target CPU/GPU vector units and would not use HMX or
HVX. We already have hand-written HVX/HMX kernels at 29 TOPS int4, and no Hexagon target exists to
carry MAX kernels anyway.

### D. Do nothing; make `lower.py` more robust
This is the baseline to compare against. It costs one pattern per new architecture, and gains
nothing from MAX.

## 6. Risks, ranked

1. **Static vs paged KV** (option A's core question): can the Llama3 and Qwen3 graphs be expressed
   or converted with a static cache without losing the zoo's benefit?
2. **IR stability** across MAX releases.
3. **Numerics parity**: hexagon_torch checks results against PyTorch, bit-exact or within 1 ulp.
   MAX's reference ops would need to become the reference, or we keep the torch reference alongside.
4. **Duplicated models**: if we write our own `max.nn` decoder, we have two model definitions again
   (`reference/` for torch, one for MAX).
5. **Board-side MAX** (only for B).

## 7. Proposed plan

| Step | Work | Gate |
|---|---|---|
| S0 | **Done here**: MAX 26.6 installed, graph builds, IR prints, no Hexagon target | the facts in sections 3 and 4 |
| S1 | **Done** (Llama3 and Qwen3, tiny config): `probe_graphs/`. ~15 op kinds; RMSNorm, QK-norm, RoPE+KV store and attention are single named ops; paged KV maps to a static cache at batch 1. | an op coverage table: **passed** |
| S2 | **Done** (Llama3 block): `mojo.max/`. A ~320-line converter over the MO ops; block lowers to exactly `lower.py`'s op multiset; 100% bit-equal to `lower.py` with the reference cos/sin table (92% with MAX's own fp32 table, max diff 2e-3). | block equals `lower.py`'s on `test_lower_block.py`: **passed** (emulator) |
| S3 | **Done**: TinyLlama, 22 layers, fp16 weights. Host emulator: 12 greedy tokens identical to the fp32 PyTorch reference (S3a). On vq: the container built from the MAX programs gives the same 12/12 tokens as the reference and as the stock container, at the same speed (~88 ms/token); 12 of its 13 sections are byte-identical to the stock `.hxb`, the weights arena differs only in the RoPE tables (S3b). `README.md`, `results/`. | tokens identical to the current `.hxb`: **passed on the DSP** |
| S4 | Qwen3, then GGUF Q4_0 direct to `w4f16v2` | same perplexity as today (`hexagon_torch/perplexity.py`) |
| S5 | A model hexagon_torch does not support (e.g. a gemma or granite variant) | works with converter additions only; this is the payoff test |
| S6 (optional) | Option B on the board | MAX installs on aarch64; a decode call costs no more than today |

S1 and S2 need no board. A first answer on option A costs a few days.

## 8. Recommendation

Pursue **option A**, gated on S1: if the zoo's graphs reduce to a small, stable op table and static
KV is tractable, MAX removes the per-architecture pattern-matching cost and adds model coverage and
GGUF loading. If paged/ragged attention proves unmappable at S1, fall back to a hand-written
`max.nn` decoder, which gains little over today, and choose option D. Skip option C. Revisit option
B only if we want MAX's serving stack.

## 9. Open items to verify
- Does MAX 26.6 install and run on the board's aarch64 Ubuntu?
- Is a Hexagon target reachable in Mojo or MAX at all (ask Modular), or is riscv32 retargeting permanent?
- How do the modulev3 Llama3 and Qwen3 graphs express attention and KV (S1)?
- Can a graph be exported without compiling it, as a stable serialized form, or must we hold a
  live `Graph` in Python?
