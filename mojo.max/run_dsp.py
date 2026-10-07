"""Run an .hxb container on the DSP through a device server and compare its greedy tokens with a
reference token list (results/s3_ref.json: hexagon_torch's fp32 PyTorch Llama, 12 tokens).

  HVXHMX_REPO=<repo> python run_dsp.py --container /path/on/board/x.hxb --skel <sha256> --port 9872 \\
      --model <checkpoint dir> [--prompt "Qualcomm is"] [--steps 11] [--ref results/s3_ref.json]
"""
import argparse
import json
import os
import sys

import max_lower  # noqa: F401  (puts HVXHMX_REPO on sys.path)

sys.path.insert(0, os.path.join(os.environ["HVXHMX_REPO"], "research", "qwen3"))
sys.path.insert(0, os.path.join(os.environ["HVXHMX_REPO"], "research", "w4a16"))
from server_qwen3 import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--container", required=True, help="path of the .hxb on the board")
ap.add_argument("--skel", required=True)
ap.add_argument("--model", required=True, help="checkpoint directory (tokenizer)")
ap.add_argument("--prompt", default="Qualcomm is")
ap.add_argument("--steps", type=int, default=11)
ap.add_argument("--host", default="10.168.168.32")
ap.add_argument("--port", type=int, default=9872)
ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "s3_ref.json"))
ap.add_argument("--out", default=None)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(a.prompt).input_ids
load, ttft, med, toks = run(a.host, a.port, a.container, ids, a.steps, a.skel)
print(f"load {load:.1f} s, prompt {len(ids)} tokens: TTFT {ttft:.1f} ms, decode median {med:.1f} ms ({1000 / med:.1f} tok/s)")
print("text  :", repr(tok.decode(ids + toks)))
print("tokens:", toks)
ref = json.load(open(a.ref))["tokens"] if os.path.exists(a.ref) else None
if ref is not None:
    n = min(len(ref), len(toks))
    same = [x == y for x, y in zip(toks, ref)]
    print(f"reference tokens: {ref}")
    print(f"identical: {sum(same)}/{n}" + ("" if all(same) else f"; first difference at index {same.index(False)}"))
if a.out:
    json.dump({"prompt_ids": ids, "tokens": toks, "ttft_ms": ttft, "decode_median_ms": med}, open(a.out, "w"))
