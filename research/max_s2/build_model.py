"""The whole Llama3 model as a MAX graph (max.pipelines' legacy `Llama3`: embedding, layers, last-token
head) from a Hugging Face config.json, with weights served lazily from safetensors by HF name
(MAX's names are the HF names without the `model.` prefix)."""
import json
import os

import numpy as np
import torch
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.kv_cache.cache_params import MHAKVCacheParams
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.llama3.model_config import Llama3Config


class LazyHF:
    """{MAX weight name: float32 ndarray}, read from the checkpoint's safetensors on access."""

    def __init__(self, path):
        from safetensors import safe_open
        self.f = safe_open(os.path.join(path, "model.safetensors"), framework="pt")

    def __getitem__(self, name):
        key = name if name == "lm_head.weight" else "model." + name
        return self.f.get_tensor(key).float().numpy()


def build(path, max_seq=2048, n_layers=None):
    hf = json.load(open(os.path.join(path, "config.json")))
    n_layers = n_layers or hf["num_hidden_layers"]
    d = DeviceRef.CPU()
    hd = hf["hidden_size"] // hf["num_attention_heads"]
    kv = MHAKVCacheParams(dtype=DType.float32, n_kv_heads=hf["num_key_value_heads"], head_dim=hd,
                          num_layers=n_layers, devices=[d])
    cfg = Llama3Config(
        hidden_size=hf["hidden_size"], num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"], num_hidden_layers=n_layers, rope_theta=hf["rope_theta"],
        rope_scaling_params=None, max_seq_len=max_seq, intermediate_size=hf["intermediate_size"],
        interleaved_rope_weights=False, vocab_size=hf["vocab_size"], dtype=DType.float32,
        model_quantization_encoding=None, quantization_config=None, kv_params=kv, rms_norm_eps=hf["rms_norm_eps"],
        attention_multiplier=hd ** -0.5, embedding_multiplier=1.0, residual_multiplier=1.0, devices=[d],
        clip_qkv=None, tie_word_embeddings=hf.get("tie_word_embeddings", False), use_subgraphs=False)
    model = Llama3(cfg)
    # the graph only needs names, shapes and dtypes here: zero-memory broadcasts stand in for the weights
    sd = {k: np.broadcast_to(np.float32(0), [int(x) for x in w.shape])
          for k, w in model.raw_state_dict().items() if "rope" not in k}
    model.load_state_dict(sd, override_quantization_encoding=True, weight_alignment=1, strict=False)
    with Graph("llama3", input_types=model.input_types(kv)) as g:
        tokens, rows, n_logits, *rest = g.inputs
        kvc = kv.unflatten_kv_inputs(iter(rest)).inputs
        g.output(*model(tokens.tensor, kvc[0], n_logits.tensor, rows.tensor))
    return g, LazyHF(path)
