"""Independent NumPy GPT-2 (float32), to check the Mojo programs against.

Reads gpt2/model.safetensors directly (no torch/transformers needed).

Usage (from ~/fun):
    uv run python tests/reference.py logits 15496,11,616,1438,318
        prints the top-5 next-token ids and logits after the given tokens
    uv run --with tiktoken python tests/reference.py greedy 464,2068,7586 30
        greedy-decodes 30 tokens (full recompute each step, no KV cache)
        and prints the decoded text with repr()
    uv run --with tiktoken python tests/reference.py ppl tests/data/alice_ch1.txt
        perplexity of a text file, with the same sliding windows as
        `gpt2t_bin --ppl` (1024 tokens, stride 512, each token scored once)

Get token ids for a prompt with:  ./gpt2t_bin -v -n 0 "your prompt"
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "gpt2" / "model.safetensors"

with open(MODEL, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    HEADER = json.loads(f.read(n))
BASE = 8 + n
BUF = np.memmap(MODEL, dtype=np.uint8, mode="r")


def t(name):
    m = HEADER[name]
    a, b = m["data_offsets"]
    return np.frombuffer(BUF[BASE + a : BASE + b], dtype=np.float32).reshape(m["shape"])


def ln(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * w + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def forward(toks, all_positions=False):
    """Returns the logits for the token after `toks`, or with all_positions,
    the logits at every position ([T, V]; row t predicts toks[t + 1])."""
    T = len(toks)
    x = t("wte.weight")[toks] + t("wpe.weight")[:T]
    for l in range(12):
        p = f"h.{l}."
        a = ln(x, t(p + "ln_1.weight"), t(p + "ln_1.bias")) @ t(p + "attn.c_attn.weight")
        a = a + t(p + "attn.c_attn.bias")
        q, k, v = (z.reshape(T, 12, 64).transpose(1, 0, 2) for z in np.split(a, 3, axis=-1))
        s = q @ k.transpose(0, 2, 1) / 8.0
        s = s + np.triu(np.full((T, T), -1e10, dtype=np.float32), 1)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, 768)
        x = x + o @ t(p + "attn.c_proj.weight") + t(p + "attn.c_proj.bias")
        h = ln(x, t(p + "ln_2.weight"), t(p + "ln_2.bias")) @ t(p + "mlp.c_fc.weight")
        m = gelu(h + t(p + "mlp.c_fc.bias"))
        x = x + m @ t(p + "mlp.c_proj.weight") + t(p + "mlp.c_proj.bias")
    x = ln(x, t("ln_f.weight"), t("ln_f.bias"))
    return (x if all_positions else x[-1]) @ t("wte.weight").T


def perplexity(ids, window=1024, stride=512):
    """Sliding-window perplexity, scoring each token once (as gpt2t does)."""
    nll, scored, scored_until, begin = 0.0, 0, 0, 0
    while True:
        end = min(begin + window, len(ids))
        logits = forward(ids[begin:end], all_positions=True).astype(np.float64)
        mx = logits.max(-1, keepdims=True)
        logp = logits - mx - np.log(np.exp(logits - mx).sum(-1, keepdims=True))
        for r in range(max(0, scored_until - begin), end - begin - 1):
            nll -= logp[r, ids[begin + r + 1]]
            scored += 1
        scored_until = end - 1
        if end == len(ids):
            break
        begin += stride
    return float(np.exp(nll / scored)), scored


def main():
    mode = sys.argv[1]
    if mode == "ppl":
        import tiktoken

        ids = tiktoken.get_encoding("gpt2").encode(open(sys.argv[2]).read())
        ppl, scored = perplexity(ids)
        print(f"perplexity {ppl:.6f} over {scored} tokens")
        return
    toks = [int(s) for s in sys.argv[2].split(",")]
    if mode == "logits":
        logits = forward(toks)
        for i in np.argsort(-logits)[:5]:
            print(i, logits[i])
    elif mode == "greedy":
        import tiktoken

        for _ in range(int(sys.argv[3])):
            toks.append(int(np.argmax(forward(toks))))
        print(repr(tiktoken.get_encoding("gpt2").decode(toks)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
