"""The PyTorch fp32 reference: hexagon_torch's own model (models/: Llama-family or Qwen3) greedy decode."""
import json
import sys

import torch

import max_lower  # noqa: F401  (puts HVXHMX_REPO on sys.path)
from hexagon_torch import models

path, prompt, n_new, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
ids = tok(prompt, return_tensors="pt").input_ids
model = models.load(path, max_seq=256).eval()          # Llama-family or Qwen3
toks, logits, pos, cur = [], [], 0, ids
with torch.no_grad():
    for _ in range(n_new):
        lg = model(cur, pos)[0, -1]
        toks.append(int(lg.argmax()))
        logits.append(lg.clone())
        pos += cur.shape[1]
        cur = torch.tensor([[toks[-1]]])
print("text:", tok.decode(ids[0].tolist() + toks))
json.dump({"prompt_ids": ids[0].tolist(), "tokens": toks, "top5": [l.topk(5).indices.tolist() for l in logits]}, open(out, "w"))
torch.save(torch.stack(logits), out + ".logits.pt")
