"""Does MAX's own fp32 RoPE table change Qwen3-0.6B's perplexity? CPU, fp32 arithmetic, WikiText-2.

  HVXHMX_REPO=<repo> python ppl_cpu.py <Qwen3-0.6B dir> [--window 512] [--windows 100] [--out results/q3_ppl_cpu.json]

Three variants on the same windows (hexagon_torch.perplexity's protocol: the test split joined with "\\n\\n",
non-overlapping windows from empty caches, every token but a window's first a target):
  fp32      the reference model;
  w4_ref    the W4 recipe (Q4_0 with llama.cpp's scales, expanded exactly as the DSP does, q4_0.dequantize_f16;
            k_proj, v_proj and layer 2's down_proj kept fp32; the output layer Q4_0) with the reference RoPE tables;
  w4_max    the same weights with the RoPE tables MAX computes (max_lower folds them from the MAX graph, fp32).
Per-window NLLs are kept, so the variants are compared pairwise (mean difference and its standard error).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import build_model
import max_lower

from hexagon_torch import lower, q4_0
from hexagon_torch.models import qwen3
from hexagon_torch.perplexity import windows as wins, wikitext2_ids

TYPES = {"q": "attn.q_proj", "k": "attn.k_proj", "v": "attn.v_proj", "o": "attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
KEEP = ("k", "v", "L2.down")


def w4(w):
    d, q = q4_0.quantize(w.numpy())
    return torch.from_numpy(q4_0.dequantize_f16(d, q).astype(np.float32))


def max_tables(path, max_seq):
    """MAX's own cos / sin tables ([max_seq, head_dim / 2], fp32), as max_lower folds them."""
    graph, w = build_model.build_layers(path, 0, 1, max_seq)
    gm, _ = max_lower.convert(graph, w, t=1, pos=0, max_seq=max_seq, batched=True, exact_tables=False)
    b = dict(gm.named_buffers())
    cos = next(v for k, v in b.items() if k.startswith("max_cos"))
    sin = next(v for k, v in b.items() if k.startswith("max_sin"))
    return cos.clone(), sin.clone()


def score(model, ids_windows):
    out = []
    for ids in ids_windows:
        model.reset()
        with torch.no_grad():
            lp = torch.log_softmax(model(ids, 0)[0, :-1].float(), -1)
        out.append((-float(lp.gather(1, ids[0, 1:, None]).sum()), ids.shape[-1] - 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    torch.set_grad_enabled(False)
    ids = wikitext2_ids(AutoTokenizer.from_pretrained(a.model))
    ws = [ids[:, s:e] for s, e in wins(ids.shape[-1], a.window)[:a.windows]]
    print(f"{ids.shape[-1]} tokens, {len(ws)} windows of {a.window}", flush=True)
    m = qwen3.load(a.model, max_seq=a.window)
    res = {}
    t0 = time.time()
    res["fp32"] = score(m, ws)
    print(f"fp32: ppl {math.exp(sum(x for x, _ in res['fp32']) / sum(n for _, n in res['fp32'])):.4f} ({time.time() - t0:.0f} s)", flush=True)

    # the W4 recipe, in place
    pol = lower.WeightPolicy("w4f16v2", KEEP)
    for i, blk in enumerate(m.layers):
        for kind, path in TYPES.items():
            if pol(f"blks.{i}.{path.split('.')[0]}.{kind}_proj.weight") != "f16":
                mod = blk.get_submodule(path)
                mod.weight.data = w4(mod.weight.data)
    head = m.lm_head
    head.weight = torch.nn.Parameter(w4(head.weight.data), requires_grad=False)   # (tied: only the output layer's copy)
    t0 = time.time()
    res["w4_ref"] = score(m, ws)
    print(f"w4_ref: ppl {math.exp(sum(x for x, _ in res['w4_ref']) / sum(n for _, n in res['w4_ref'])):.4f} ({time.time() - t0:.0f} s)", flush=True)

    cos, sin = max_tables(a.model, a.window)
    ref = [(blk.attn.cos.clone(), blk.attn.sin.clone()) for blk in m.layers[:1]][0]
    print(f"MAX table vs reference: max |cos diff| {(cos.double() - ref[0]).abs().max():.2e}, "
          f"max |sin diff| {(sin.double() - ref[1]).abs().max():.2e}", flush=True)
    for blk in m.layers:
        blk.attn.cos.copy_(cos.double())
        blk.attn.sin.copy_(sin.double())
    t0 = time.time()
    res["w4_max"] = score(m, ws)
    print(f"w4_max: ppl {math.exp(sum(x for x, _ in res['w4_max']) / sum(n for _, n in res['w4_max'])):.4f} ({time.time() - t0:.0f} s)", flush=True)

    ppl = lambda r: math.exp(sum(x for x, _ in r) / sum(n for _, n in r))
    d = np.array([x for x, _ in res["w4_max"]]) - np.array([x for x, _ in res["w4_ref"]])
    n = sum(n for _, n in res["w4_ref"])
    print(f"\nw4_max - w4_ref: total NLL difference {d.sum():+.4f} nats over {n} targets "
          f"({d.sum() / n:+.2e} per token), per-window mean {d.mean():+.4f} +- {d.std(ddof=1) / math.sqrt(len(d)):.4f}; "
          f"ppl ratio {ppl(res['w4_max']) / ppl(res['w4_ref']):.6f}")
    print(f"w4_ref vs fp32: ppl {ppl(res['w4_ref']):.4f} vs {ppl(res['fp32']):.4f} ({100 * (ppl(res['w4_ref']) / ppl(res['fp32']) - 1):+.2f}%)")
    if a.out:
        json.dump({"window": a.window, "windows": len(ws), "nll_n": res,
                   "ppl": {k: ppl(v) for k, v in res.items()}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
