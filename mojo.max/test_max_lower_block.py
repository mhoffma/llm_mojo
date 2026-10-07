"""S2 gate: the MAX Llama3 block, converted to hexagon:: ops, matches hexagon_torch's own lowering
of the reference block (tests/test_lower_block.py) and the PyTorch reference.

HVXHMX_REPO=<hvxhmx_mojo checkout> python -m pytest -q test_max_lower_block.py
(needs torch + max==26.6.0 + the max.pipelines dependencies; see probe_graphs/README.md)"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import max_lower  # noqa: E402  (puts HVXHMX_REPO on sys.path)
from build_block import build  # noqa: E402
from hexagon_torch import layouts, lower  # noqa: E402
from hexagon_torch.reference import init_  # noqa: E402
from hexagon_torch.reference.modules import Config, DecoderBlock  # noqa: E402

CFG = Config(hidden=512, n_heads=8, n_kv_heads=4, ffn=1408, max_seq=256)     # head_dim 64, as test_lower_block
T = 64


class WithCache(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blk = init_(DecoderBlock(cfg), seed=1).eval()
        self.register_buffer("k", torch.zeros(1, cfg.n_kv_heads, cfg.max_seq, cfg.head_dim))
        self.register_buffer("v", torch.zeros(1, cfg.n_kv_heads, cfg.max_seq, cfg.head_dim))

    def forward(self, x):
        return self.blk(x, {"k": self.k, "v": self.v}, 0)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    x = torch.randn(1, T, CFG.hidden)
    mod = WithCache(CFG)
    ref_mod = WithCache(CFG)
    ref_mod.load_state_dict(mod.state_dict())
    ref = ref_mod(x)
    low = lower.lower(torch.export.export(mod, (x,)))             # hexagon_torch's own path
    state = {k.removeprefix("blk."): v.detach().numpy() for k, v in mod.state_dict().items() if k.startswith("blk.")}
    graph, weights = build(CFG, state)
    gm, counts = max_lower.convert(graph, weights, t=T, pos=0, max_seq=CFG.max_seq)
    return x, ref, ref_mod, low, gm, counts


def test_op_counts_match_lower_py(setup):
    _, _, _, low, _, counts = setup
    theirs = {k: v for k, v in lower.count_ops(low).items() if k.startswith("hexagon.")}
    mine = {k.replace("hexagon.", "hexagon.", 1) + ".default" if not k.endswith(".default") else k: v
            for k, v in counts.items()}
    norm = lambda d: {k.removeprefix("hexagon::").removeprefix("hexagon.").removesuffix(".default")
                      .removesuffix(".default"): v for k, v in d.items()}
    assert norm(mine) == norm(theirs), (norm(mine), norm(theirs))


def test_matches_hexagon_torch_lowering(setup):
    x, _, _, low, gm, _ = setup
    y_lower = low.module()(x)[0]
    y_max = gm(x[0])
    d = (y_max - y_lower).abs()
    print(f"\nvs lower.py: max |diff| {d.max():.3e}, exactly equal {(d == 0).float().mean():.4f}")
    torch.testing.assert_close(y_max, y_lower, rtol=2e-2, atol=2e-2)


def test_matches_pytorch_reference_and_cache(setup):
    x, ref, ref_mod, _, gm, _ = setup
    y = gm(x[0])
    torch.testing.assert_close(y, ref[0], rtol=2e-2, atol=2e-2)
    bufs = dict(gm.named_buffers())
    kc, vc = bufs[gm.caches["k"]].numpy(), bufs[gm.caches["v"]].numpy()
    hd, n_kv, s = CFG.head_dim, CFG.n_kv_heads, CFG.max_seq
    k = layouts.unpack_kcache(kc, n_kv, s, hd)[:, :T]
    v = layouts.unpack_vcache(vc, n_kv, s, hd)[:, :T]
    np.testing.assert_allclose(k, ref_mod.k[0, :, :T].detach().numpy(), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(v, ref_mod.v[0, :, :T].detach().numpy(), rtol=2e-2, atol=2e-2)
    assert not layouts.unpack_kcache(kc, n_kv, s, hd)[:, T:].any()
