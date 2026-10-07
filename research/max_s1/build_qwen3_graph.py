import sys, numpy as np
from max.dtype import DType
from max.graph import Graph, DeviceRef
from max.nn.kv_cache.cache_params import MHAKVCacheParams
from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
import max.nn.comm.allreduce as _ar
def _init(self,num_accelerators=1,**k):
    from max.nn.layer import Module; Module.__init__(self); self.devices=[]
_ar.Allreduce.__init__=_init   # single device: never called; avoids opening a GPU
import max.pipelines.architectures.qwen3.qwen3 as _q; _q.Allreduce=_ar.Allreduce
d=DeviceRef.CPU()
kv=MHAKVCacheParams(dtype=DType.float32, n_kv_heads=2, head_dim=16, num_layers=2, devices=[d])
cfg=Qwen3Config(hidden_size=64,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2,
  rope_theta=1000000.0,rope_scaling_params=None,max_seq_len=128,intermediate_size=128,
  interleaved_rope_weights=False,vocab_size=256,dtype=DType.float32,model_quantization_encoding=None,
  quantization_config=None,kv_params=kv,rms_norm_eps=1e-6,attention_multiplier=16**-0.5,
  embedding_multiplier=1.0,residual_multiplier=1.0,devices=[d],clip_qkv=None,
  tie_word_embeddings=True,use_subgraphs=False)
m=Qwen3(cfg)
raw=m.raw_state_dict()
print(len(raw),'weights'); print(*sorted(raw)[:30],sep='\n')
sd={k:np.zeros([int(x) for x in w.shape],dtype=np.float32) for k,w in raw.items() if 'rope' not in k}
m.load_state_dict(sd,override_quantization_encoding=True,weight_alignment=1,strict=False)
with Graph("qwen3",input_types=m.input_types(kv)) as g:
    tokens,rows,nl,*rest=g.inputs
    sig=[v.buffer for v in rest[:1]]
    kvc=kv.unflatten_kv_inputs(iter(rest[1:])).inputs
    out=m(tokens.tensor,kvc,nl.tensor,rows.tensor,sig)
    g.output(*out)
open(sys.argv[1],"w").write(g.module._to_mlir_str())
