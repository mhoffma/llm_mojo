"""S3a: greedy decode of TinyLlama through the MAX front end, on the hexagon_torch host emulator.

  HVXHMX_REPO=<repo> python run_tinyllama.py <checkpoint dir> "prompt" [new tokens] [layers] [out.json]

MAX graph (max.pipelines Llama3) -> max_lower.convert -> one program of hexagon:: ops with runtime
position, used for the prompt (one call of len(prompt) rows) and for every decode step (one row).
"""
import json
import sys
import time

import torch

import build_model
import max_lower


def main():
    path, prompt = sys.argv[1], sys.argv[2]
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    out = sys.argv[5] if len(sys.argv) > 5 else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    max_seq = 256
    assert len(ids) + n_new <= max_seq
    t0 = time.time()
    graph, weights = build_model.build(path, max_seq=max_seq, n_layers=layers)
    t1 = time.time()
    gm, counts = max_lower.convert(graph, weights, max_seq=max_seq)
    t2 = time.time()
    print(f"graph {t1 - t0:.1f} s, convert+pack {t2 - t1:.1f} s; ops {dict(sorted(counts.items()))}", flush=True)
    toks, logits = [], []
    pos, cur = 0, ids
    with torch.no_grad():
        for step in range(n_new):
            t0 = time.time()
            lg = gm(cur, pos)[0]                      # [vocab], the last row
            nxt = int(lg.argmax())
            logits.append(lg.clone())
            toks.append(nxt)
            pos += len(cur)
            cur = torch.tensor([nxt])
            print(f"step {step}: token {nxt} {tok.decode([nxt])!r} ({time.time() - t0:.1f} s)", flush=True)
            if nxt == tok.eos_token_id:
                break
    print("text:", tok.decode(ids.tolist() + toks))
    if out:
        json.dump({"prompt_ids": ids.tolist(), "tokens": toks, "top5": [l.topk(5).indices.tolist() for l in logits],
                   "layers": layers}, open(out, "w"))
        torch.save(torch.stack(logits), out + ".logits.pt")


if __name__ == "__main__":
    main()
