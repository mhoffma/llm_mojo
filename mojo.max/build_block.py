"""Build a MAX graph of one Llama3 decoder block (the legacy `max.nn` TransformerBlock, as
max.pipelines assembles it) with given weights. Input: hidden states [total_seq_len, hidden] and
the paged-KV inputs; output: the block's hidden states."""
import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache.cache_params import MHAKVCacheParams
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.llama3.model_config import Llama3Config

NAMES = {   # hexagon_torch reference DecoderBlock parameter -> MAX weight name (layer 0)
    "attn_norm.weight": "layers.0.input_layernorm.weight",
    "mlp_norm.weight": "layers.0.post_attention_layernorm.weight",
    **{f"attn.{p}_proj.weight": f"layers.0.self_attn.{p}_proj.weight" for p in "qkvo"},
    **{f"mlp.{p}_proj.weight": f"layers.0.mlp.{p}_proj.weight" for p in ("gate", "up", "down")},
}


def build(cfg, ref_state):
    """cfg: hexagon_torch.reference.modules.Config; ref_state: {reference name: ndarray}.
    Returns (Graph, {MAX weight name: ndarray})."""
    d = DeviceRef.CPU()
    kv = MHAKVCacheParams(dtype=DType.float32, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
                          num_layers=1, devices=[d])
    mc = Llama3Config(
        hidden_size=cfg.hidden, num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads,
        num_hidden_layers=1, rope_theta=cfg.rope_theta, rope_scaling_params=None, max_seq_len=cfg.max_seq,
        intermediate_size=cfg.ffn, interleaved_rope_weights=False, vocab_size=32, dtype=DType.float32,
        model_quantization_encoding=None, quantization_config=None, kv_params=kv, rms_norm_eps=cfg.rms_eps,
        attention_multiplier=cfg.head_dim ** -0.5, embedding_multiplier=1.0, residual_multiplier=1.0,
        devices=[d], clip_qkv=None, use_subgraphs=False)
    model = Llama3(mc)
    raw = model.raw_state_dict()
    weights = {NAMES[k]: np.asarray(v, np.float32) for k, v in ref_state.items() if k in NAMES}
    for k, w in raw.items():                                  # the unused ones (embedding, head, norm)
        weights.setdefault(k, np.zeros([int(x) for x in w.shape], np.float32))
    model.load_state_dict({k: v for k, v in weights.items() if "rope" not in k},
                          override_quantization_encoding=True, weight_alignment=1, strict=False)
    types = model.input_types(kv)
    hidden = TensorType(DType.float32, ["total_seq_len", cfg.hidden], d)
    with Graph("block", input_types=[hidden, *types[1:]]) as g:       # [hidden, row_offsets, n_logits, kv...]
        h, rows, _n_logits, *kv_in = g.inputs
        kvc = kv.unflatten_kv_inputs(iter(kv_in)).inputs[0]
        from max.graph import ops
        out = model.layers[0](ops.constant(0, DType.uint32, device=DeviceRef.CPU()), h.tensor, kvc,
                              freqs_cis=model.rope.freqs_cis, input_row_offsets=rows.tensor)
        g.output(out)
    return g, weights
