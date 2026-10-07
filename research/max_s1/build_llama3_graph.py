import re, collections, sys
from max.dtype import DType
from max.graph import Graph, DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.cache_params import MHAKVCacheParams
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.llama3.model_config import Llama3Config
d=DeviceRef.CPU()
kv=MHAKVCacheParams(dtype=DType.float32, n_kv_heads=2, head_dim=16, num_layers=2, devices=[d])
cfg=Llama3Config(hidden_size=64,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2,
  rope_theta=10000.0,rope_scaling_params=None,max_seq_len=128,intermediate_size=128,
  interleaved_rope_weights=False,vocab_size=256,dtype=DType.float32,model_quantization_encoding=None,
  quantization_config=None,kv_params=kv,rms_norm_eps=1e-5,attention_multiplier=16**-0.5,
  embedding_multiplier=1.0,residual_multiplier=1.0,devices=[d],clip_qkv=None,use_subgraphs=False)
m=Llama3(cfg)
import numpy as np
raw=m.raw_state_dict()
print(len(raw),'weights', list(raw)[:6])
sd={k:np.zeros([int(x) for x in w.shape],dtype=np.float32) for k,w in raw.items() if 'rope' not in k}
m.load_state_dict(sd,override_quantization_encoding=True,weight_alignment=1,strict=False)
with Graph("llama3",input_types=m.input_types(kv)) as g:
    tokens,rows,nl,*rest=g.inputs
    kvc=kv.unflatten_kv_inputs(iter(rest)).inputs
    out=m(tokens.tensor,kvc[0],nl.tensor,rows.tensor)
    g.output(*out)
txt=g.module._to_mlir_str()
open(sys.argv[1],"w").write(txt)
