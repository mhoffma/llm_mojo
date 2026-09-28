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

Add --quant BITS,GROUP,SYM (e.g. --quant 4,32,0) to any mode to fake-quantize
the same matrices gpt2t quantizes (the four per-layer matrices and wte) with
the same scheme as tensor.mojo's QuantMatrix: round-to-nearest, float16 scale
and integer zero point per group along the reduction axis, GROUP 0 =
per-channel, SYM 1 = symmetric. Use it to check `gpt2t_bin --dtype int...` independently.

Add --gguf FILE to any mode to take the weights from a llama.cpp GGUF file
instead (dequantized to float32 with the `gguf` package, so add
`--with gguf` to uv run). GGUF stores GPT-2's output head as a separate
tensor (output.weight), which is used in place of the tied wte.
Add --quant-only NAME (e.g. mlp.c_proj, or wte) to quantize only the matrices
whose name ends with NAME, to find which ones lose the most accuracy.

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
QUANT = None  # (bits, group, symmetric) when --quant is given
QUANTIZED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight",
             "mlp.c_proj.weight", "wte.weight")
_cache = {}
GGUF = None  # a gguf.GGUFReader when --gguf is given


def gguf_name(name):
    """Maps a Hugging Face GPT-2 tensor name to its llama.cpp GGUF name."""
    fixed = {"wte.weight": "token_embd.weight", "wpe.weight": "position_embd.weight",
             "ln_f.weight": "output_norm.weight", "ln_f.bias": "output_norm.bias",
             "lm_head": "output.weight"}
    if name in fixed:
        return fixed[name]
    _, layer, rest = name.split(".", 2)
    parts = {"ln_1": "attn_norm", "attn.c_attn": "attn_qkv", "attn.c_proj": "attn_output",
             "ln_2": "ffn_norm", "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down"}
    module, kind = rest.rsplit(".", 1)
    return f"blk.{layer}.{parts[module]}.{kind}"


def t_gguf(name):
    from gguf.quants import dequantize

    want = gguf_name(name)
    tensor = next((x for x in GGUF.tensors if x.name == want), None)
    if tensor is None and name == "lm_head":  # tied: no separate output head
        return t("wte.weight")
    w = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    w = w.reshape([int(d) for d in reversed(tensor.shape)])
    # GGUF stores layer matrices as [OUT, IN]; Hugging Face's Conv1D is [IN, OUT].
    if w.ndim == 2 and name.startswith("h.") and name.endswith(".weight"):
        w = w.T
    return w


def fake_quant(w, bits, group, sym, reduce_rows):
    """Quantizes and dequantizes w like tensor.mojo's QuantMatrix:
    w = scale * (u - zero_point), float16 scale and integer zero point per
    group along the reduction axis."""
    W = w if reduce_rows else w.T  # put the reduction axis first
    rows, cols = W.shape
    g = rows if group == 0 else group
    Wg = W.reshape(rows // g, g, cols)
    levels = 2**bits - 1
    lo = np.minimum(Wg.min(axis=1), 0)  # the range always includes 0
    hi = np.maximum(Wg.max(axis=1), 0)
    if sym:
        s = np.maximum(-lo, hi) / (2 ** (bits - 1) - 1)
    else:
        s = (hi - lo) / levels
    s = np.where(s == 0, 1, s).astype(np.float16).astype(np.float32)
    if sym:
        zp = np.full_like(s, 2 ** (bits - 1))
    else:
        zp = np.clip(np.round(-lo / s), 0, levels)
    s, zp = s[:, None, :], zp[:, None, :]
    u = np.clip(np.round(Wg / s) + zp, 0, levels)
    out = ((u - zp) * s).reshape(rows, cols).astype(np.float32)
    return out if reduce_rows else out.T


def t(name):
    if name in _cache:
        return _cache[name]
    if GGUF is not None:
        _cache[name] = t_gguf(name)
        return _cache[name]
    m = HEADER[name]
    a, b = m["data_offsets"]
    w = np.frombuffer(BUF[BASE + a : BASE + b], dtype=np.float32).reshape(m["shape"])
    if QUANT and name.endswith(QUANTIZED):
        w = fake_quant(w, *QUANT, reduce_rows=(name != "wte.weight"))
        _cache[name] = w
    return w


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
    head = t("lm_head") if GGUF is not None else t("wte.weight")
    return (x if all_positions else x[-1]) @ head.T


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
    global QUANT, GGUF
    if "--gguf" in sys.argv:
        from gguf import GGUFReader

        i = sys.argv.index("--gguf")
        GGUF = GGUFReader(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    if "--quant" in sys.argv:
        i = sys.argv.index("--quant")
        bits, group, sym = (int(v) for v in sys.argv[i + 1].split(","))
        QUANT = (bits, group, bool(sym))
        del sys.argv[i : i + 2]
    if "--quant-only" in sys.argv:
        global QUANTIZED
        i = sys.argv.index("--quant-only")
        QUANTIZED = tuple(q for q in QUANTIZED if q.startswith(sys.argv[i + 1]))
        del sys.argv[i : i + 2]
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
