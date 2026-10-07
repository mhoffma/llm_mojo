"""Perplexity of Qwen3-0.6B (the W4 recipe) on the DSP with the stock programs or the MAX-made ones.

  HVXHMX_REPO=<repo> python ppl_dsp.py <Qwen3-0.6B dir> --mode stock|max|maxown --skel <sha256> \\
      --remote tcp://10.168.168.32:9872 --lib-dir <dir on vq> [--window 512] [--windows 40] [--out x.json]

  stock   hexagon_torch's own front end (torch.export + lower.py), the stock Generator;
  max     MaxGenerator with the reference's RoPE tables (the default of the MAX path);
  maxown  MaxGenerator with MAX's own fp32 RoPE tables.
The Generator is built in a remote Session (programs, uploads and schedules, as export_blob does) and scores
WikiText-2 windows with Generator.nll (prefill chunks, the head over every position), hexagon_torch.perplexity's
protocol. Per-window (NLL, targets) are written for paired comparison.
"""
import argparse
import json
import math
import time

import torch

import max_lower  # noqa: F401  (puts HVXHMX_REPO on sys.path)

from hexagon_torch import generate, lower, models, session
from hexagon_torch.perplexity import windows as wins, wikitext2_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", required=True, choices=["stock", "max", "maxown"])
    ap.add_argument("--skel", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--prefill-rows", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    torch.set_grad_enabled(False)
    ids = wikitext2_ids(AutoTokenizer.from_pretrained(a.model))
    ws = [ids[:, s:e] for s, e in wins(ids.shape[-1], a.window)[:a.windows]]
    m = models.load(a.model, max_seq=a.window, dtype=torch.bfloat16)
    pol = lower.WeightPolicy("w4f16v2", ("k", "v", "L2.down"), "ref")
    Gen = generate.Generator
    if a.mode != "stock":
        import gen_max
        import os
        gen_max.MaxGenerator.checkpoint = os.path.abspath(a.model)
        gen_max.MaxGenerator.exact_tables = a.mode == "max"
        Gen = gen_max.MaxGenerator
    res = []
    with lower.pack_cache():
        with session.Session.waiting(skel=a.skel, remote_addr=a.remote, lib_dir=a.lib_dir) as s:
            s.arena_max = 1 << 40
            t0 = time.time()
            gen = Gen(s, m, context=a.window, log=None, weights=pol, head_weights="w4f16v2",
                      prefill_rows=a.prefill_rows, head_rows=a.prefill_rows, decode=False)
            print(f"[{a.mode}] session open, programs built in {time.time() - t0:.0f} s", flush=True)
            t0 = time.time()
            for i, w in enumerate(ws):
                gen.reset()
                nll, n = gen.nll(w)
                res.append((nll, n))
                if (i + 1) % 5 == 0 or i + 1 == len(ws):
                    tot, cnt = sum(x for x, _ in res), sum(c for _, c in res)
                    print(f"[{a.mode}] window {i + 1}/{len(ws)}: ppl so far {math.exp(tot / cnt):.4f} ({time.time() - t0:.0f} s)", flush=True)
    tot, cnt = sum(x for x, _ in res), sum(c for _, c in res)
    print(f"[{a.mode}] ppl {math.exp(tot / cnt):.4f} over {cnt} targets")
    if a.out:
        json.dump({"mode": a.mode, "window": a.window, "nll_n": res, "ppl": math.exp(tot / cnt)}, open(a.out, "w"))


if __name__ == "__main__":
    main()
