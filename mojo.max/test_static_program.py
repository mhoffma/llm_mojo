"""S3b, offline gate: the static-shape program MAX makes for a group of layers is the program
lower.lower() makes for the same layers, as far as the DSP flow can tell.

Compared per row count (a prefill chunk and a decode row): the hexagon ops in order with their
scalar arguments, the constant tensors they read (packed weights, bias tables, gammas, RoPE tables,
the initial caches: byte for byte), and hexagon_torch's memory plan (peak VTCM, where every value goes).

TINYLLAMA_DIR=<checkpoint> HVXHMX_REPO=<hvxhmx_mojo> python -m pytest -q -s test_static_program.py
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import max_lower  # noqa: E402
import build_model  # noqa: E402
from hexagon_torch import lower, memplan  # noqa: E402
from hexagon_torch.models import llama  # noqa: E402

CKPT = os.environ.get("TINYLLAMA_DIR")
pytestmark = pytest.mark.skipif(not CKPT, reason="set TINYLLAMA_DIR to a TinyLlama checkpoint directory")
LAYERS, MAX_SEQ = 2, 256


@pytest.fixture(scope="module")
def model():
    return llama.load(CKPT, max_seq=MAX_SEQ, n_layers=LAYERS)


def hexagon_ops(ep):
    return [n for n in ep.graph.nodes if memplan._is_hexagon(n)]


def const(ep, node):
    """The tensor behind a get_attr / placeholder argument."""
    r = memplan._root(node)
    sig = ep.graph_signature
    key = {**sig.inputs_to_buffers, **sig.inputs_to_parameters}.get(r.name)
    if key is None:
        return None
    return (ep.state_dict[key] if key in ep.state_dict else ep.constants[key]).numpy()


def describe(ep):
    out = []
    for n in hexagon_ops(ep):
        row = [str(n.target)]
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                c = const(ep, a)
                row.append(None if c is None else ("const", c.shape, c.dtype.str, hash(c.tobytes())))
            else:
                row.append(float(np.float32(a)) if isinstance(a, float) else a)    # (MAX holds eps as fp32)
        if "scaled_dot_product_attention" in row[0] and row[10] <= 0:      # (0: the default, 1 / sqrt(head_dim))
            row[10] = float(np.float32(row[7] ** -0.5))
        out.append(row)
    return out


WEIGHTS = {"f16": lambda: "f16", "w4f16v2": lambda: "w4f16v2",
           "w4_mixed": lambda: lower.WeightPolicy("w4f16v2", keep=("k", "L1.down"))}     # (a policy per call: .at() is pure)


def policy_of(weights):
    return weights if isinstance(weights, lower.WeightPolicy) else lower.WeightPolicy(weights)


def mine(model, graph_fn, rows, weights="f16", exact_tables=True):
    graph, w = graph_fn()
    gm, _ = max_lower.convert(graph, w, t=rows, pos=0, max_seq=MAX_SEQ, batched=True, exact_tables=exact_tables,
                              policy=policy_of(weights))
    with torch.no_grad():
        return torch.export.export(gm, (torch.randn(1, rows, model.cfg.hidden),))


def theirs(model, rows, mod, weights="f16"):
    with torch.no_grad():
        return lower.lower(torch.export.export(mod, (torch.randn(1, rows, model.cfg.hidden),)), weights=weights)


@pytest.mark.parametrize("wname", list(WEIGHTS))
@pytest.mark.parametrize("rows", [32, 1])
def test_layers_program_equals_lower_py(model, rows, wname):
    weights = WEIGHTS[wname]()
    with lower.pack_cache():
        a = mine(model, lambda: build_model.build_layers(CKPT, 0, LAYERS, MAX_SEQ), rows, weights)
        b = theirs(model, rows, llama.Layers(model, 0, LAYERS), weights)
    da, db = describe(a), describe(b)
    assert len(da) == len(db), (len(da), len(db))
    for i, (x, y) in enumerate(zip(da, db)):
        assert x == y, (i, x[0], x, y)
    pa = memplan.plan_memory(a, 8 << 20, context=MAX_SEQ)
    pb = memplan.plan_memory(b, 8 << 20, context=MAX_SEQ)
    assert pa.peak == pb.peak and pa.live_peak == pb.live_peak, (pa.peak, pb.peak)
    na, nb = [n.name for n in hexagon_ops(a)], [n.name for n in hexagon_ops(b)]
    def pl(p, names):            # (kind, place, VTCM offset, bytes, shape) of each op's output region (first/last are node
        # positions, which differ: lower.py's graph carries extra cast and view nodes)
        return [None if n not in p.values else
                (lambda r: (r.kind, r.place, r.offset, r.bytes, tuple(r.shape)))(p.values[n]) for n in names]
    assert pl(pa, na) == pl(pb, nb)
    # The one deliberate difference: lower.py registers its own cos / sin tables for each RoPE call (q and k),
    # max_lower shares one pair per layer between them: 2 fewer 32 KiB arena regions per layer, same bytes.
    rope_tables = lambda p: [r for r in p.regions if r.kind == "weights" and r.place == "arena" and r.bytes == 256 * 32 * 4]
    assert len(rope_tables(pb)) - len(rope_tables(pa)) == 2 * LAYERS, (len(rope_tables(pa)), len(rope_tables(pb)))
    rest = lambda p: [(r.kind, r.place, r.offset, r.bytes) for r in p.regions if r not in rope_tables(p)]
    assert rest(pa) == rest(pb)
    print(f"\n{wname}, rows {rows}: {len(da)} hexagon ops identical, constants identical, plan peak {pa.peak} bytes of VTCM")


@pytest.mark.parametrize("wname", ["f16", "w4f16v2"])
def test_head_program_equals_lower_py(model, wname):
    rows, weights = 1, WEIGHTS[wname]()
    with lower.pack_cache():
        a = mine(model, lambda: build_model.build_head(CKPT, MAX_SEQ), rows, weights)
        b = theirs(model, rows, llama.Head(model), weights)
    assert describe(a) == describe(b)
    assert memplan.plan_memory(a, 8 << 20).peak == memplan.plan_memory(b, 8 << 20).peak
