"""S2: lower a MAX graph (one Llama3 decoder block) onto `hexagon::` ops.

Walks the MO ops of a `max.graph.Graph` in order and emits a torch.fx program of `hexagon::` ops
(hexagon_torch), the way hexagon_torch/lower.py does from an aten graph. No pattern matching over
decomposed ops: every MO op is dispatched by name.

  * constants (mo.constant, mo.constant.external) and everything computed from constants only
    (range, div, pow, cast, reshape, mul, cos, sin, concat, transpose, slice) is folded with numpy;
    that covers MAX's in-graph cos/sin table and the q|k|v and gate|up weight concatenations;
  * mo.reduce.rms_norm   -> hexagon.rms_norm
  * rmo.matmul by a constant weight -> hexagon.linear (fp16); a matmul whose result is split
    (mo.split, or the fused QKV that rope_split_store splits) becomes one linear per piece
  * mo.custom rope_split_store.ragged.paged -> hexagon.rope (q, k) + hexagon.index_copy (k, v)
  * mo.custom mha.ragged.paged -> hexagon.scaled_dot_product_attention
  * silu * up -> hexagon.swiglu;  add -> hexagon.add

Specialization: the sequence length `t`, the position `pos` and the cache length `max_seq` are
arguments (MAX keeps them symbolic / in the cache_lengths input). Batch 1, prefill of one block.
Values on the DSP are [t, features] fp16; reshapes between [t, h, d] and [t, h*d] are aliases.
"""
import os
import re
import sys

import numpy as np
import torch

HVXHMX = os.environ.get("HVXHMX_REPO")
if HVXHMX and HVXHMX not in sys.path:
    sys.path.insert(0, HVXHMX)

from hexagon_torch import layouts, lower, ops  # noqa: E402  (registers torch.ops.hexagon.*)
from hexagon_torch.pack import LAYOUT_VERSION, pack_linear_f16, pack_rms_norm  # noqa: E402

aten, H = torch.ops.aten, torch.ops.hexagon
_NP = {"f32": np.float32, "f64": np.float64, "f16": np.float16, "si64": np.int64, "ui32": np.uint32,
       "si32": np.int32, "bool": np.bool_}


# ---- reading MLIR ops ---------------------------------------------------------------------------
def _type(v):
    """'!mo.tensor<[t, 128], f32>' -> (['t', 128], 'f32')."""
    m = re.match(r"!mo\.tensor<\[(.*?)\], (\w+)>", str(v.type))
    if m is None:
        return None, None
    parts, depth, cur = [], 0, ""
    for ch in m.group(1):                                  # split on commas outside parentheses: add(n, -1)
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    dims = [int(d) if re.fullmatch(r"-?\d+", d.strip()) else d.strip() for d in parts if d.strip()]
    return dims, m.group(2)


def _attr(op, name):
    try:
        return str(op.attributes[name])
    except KeyError:
        return None


def _dense(op, shape, dt):
    nums = re.search(r"dense_array<([^>]*)>", _attr(op, "value")).group(1)
    arr = np.array([float(x) for x in nums.split(",")], dtype=np.float64).astype(_NP[dt])
    n = int(np.prod(shape)) if shape else 1
    return np.broadcast_to(arr, (n,)).reshape(shape).copy() if arr.size == 1 else arr.reshape(shape)


def _symbol(op):
    m = re.search(r'symbol = "([^"]+)"', str(op))
    return m.group(1) if m else None


def _param(op, key):
    m = re.search(rf"{key} = (\"[^\"]*\"|[^,}}\s]+)", str(op).split("}")[0] + "}")
    return m.group(1).strip('"') if m else None


# ---- folding constants --------------------------------------------------------------------------
def _fold(name, op, ins, shape, dt):
    f = {
        "rmo.mo.range": lambda: np.arange(ins[0], ins[1], ins[2], dtype=_NP[dt]),
        "rmo.div": lambda: (ins[0] / ins[1]).astype(_NP[dt]),
        "rmo.pow": lambda: np.power(ins[0], ins[1]).astype(_NP[dt]),
        "rmo.mul": lambda: (ins[0] * ins[1]).astype(_NP[dt]),
        "rmo.add": lambda: (ins[0] + ins[1]).astype(_NP[dt]),
        "rmo.sub": lambda: (ins[0] - ins[1]).astype(_NP[dt]),
        "mo.cast": lambda: ins[0].astype(_NP[dt]),
        "rmo.mo.cos": lambda: np.cos(ins[0]).astype(_NP[dt]),
        "rmo.mo.sin": lambda: np.sin(ins[0]).astype(_NP[dt]),
        "rmo.reshape": lambda: ins[0].reshape(shape),
        "rmo.mo.transpose": lambda: ins[0].T if ins[0].ndim == 2 else None,
        "rmo.concat": lambda: np.concatenate(ins, axis=int(re.search(r"axis = (-?\d+)", str(op)).group(1))),
    }.get(name)
    return None if f is None else f()


class Converter:
    def __init__(self, graph, weights, t=None, pos=None, max_seq=256, batched=False, exact_tables=False):
        """t, pos: ints bake the sequence length / position into the program (S2); None makes them
        runtime: the program is forward(x, pos) and t is the number of rows of x."""
        self.g, self.weights, self.t, self.pos, self.max_seq = graph, weights, t, pos, max_seq
        self.exact_tables = exact_tables   # fold MAX's cos/sin chain in fp64 (hexagon_torch's reference tables)
        self.batched = batched      # hidden input / output as [1, t, features] (a torch module's), t static
        self.fx = torch.fx.Graph()
        self.gm_buffers = {}
        self.val = {}               # mlir value -> ("const", ndarray) | ("lazy", name) | ("fx", node) | ...
        self.counts = {}
        self.n = 0
        self.caches = {}            # layer -> (K buffer name, V buffer name)
        self.cache_nodes = {}       # layer -> (K get_attr node, V get_attr node)
        self.uses = {}

    # -- emission helpers
    def buffer(self, name, tensor):
        name = f"max_{name}{self.n}"
        self.n += 1
        self.gm_buffers[name] = tensor
        return self.fx.get_attr(name)

    def call(self, fn, *args):
        key = str(fn).replace("hexagon.", "hexagon.")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.fx.call_function(fn, args)

    def f16(self, x):
        return self.fx.call_function(aten._to_copy.default, (x,), {"dtype": torch.float16})

    def fdt(self, dt):
        """The dtype to fold a constant in: with exact_tables, fp32 chains (MAX's in-graph cos/sin table)
        run in fp64 and round once at the consumer, as hexagon_torch's rope_tables does."""
        return "f64" if self.exact_tables and dt == "f32" else dt

    # -- lookup
    def const(self, v):
        kind, x = self.val[v]
        if kind == "lazy":
            x = np.asarray(self.weights[x])
            self.val[v] = ("const", x)
            return x
        if kind != "const":
            raise NotImplementedError(f"expected a constant, got {kind}")
        return x

    def is_const(self, v):
        return self.val[v][0] in ("const", "lazy")

    def node(self, v):
        kind, x = self.val[v]
        if kind != "fx":
            raise NotImplementedError(f"expected a DSP value, got {kind}")
        return x

    # -- the walk
    def run(self):
        body = self.g._body
        args = list(body.arguments)
        x = self.fx.placeholder("x")
        if self.pos is None:
            self.pos = self.fx.placeholder("pos")
        if self.t is None:
            self.t = self.fx.call_function(aten.sym_size.int, (x, 0))
        if _type(args[0])[1] == "si64":                       # token ids: the embedding is a host gather
            self.val[args[0]] = ("tokens", x)
        elif self.batched:
            x2 = self.fx.call_function(aten.reshape.default, (x, [self.t, _type(args[0])[0][-1]]))
            self.val[args[0]] = ("fx", self.f16(x2))
        else:
            self.val[args[0]] = ("fx", self.f16(x))
        for a in args[1:]:
            self.val[a] = ("opaque", None)           # rows, kv cache, page table...: consumed by custom ops
        self.dispatch_all(body)
        return self.finish()

    def dispatch_all(self, body):
        ops_ = [op.operation for op in body.operations]
        for o in ops_:
            for i in o.operands:
                self.uses[i] = self.uses.get(i, 0) + 1
            if o.name == "mo.output":
                self.out = o.operands[0]
        for o in ops_:
            name = o.name
            if name in ("mo.chain.create", "mo.output"):
                if name == "mo.output":
                    self.out = o.operands[0]
                continue
            res = list(o.results)
            ins = list(o.operands)
            h = getattr(self, "op_" + name.replace(".", "_"), None)
            if h is not None:
                h(o, ins, res)
            elif all(self.val.get(i, ("", 0))[0] in ("const", "lazy") for i in ins) and res:
                shape, dt = _type(res[0])
                out = _fold(name, o, [self.const(i) for i in ins], shape, self.fdt(dt))
                if out is None:
                    raise NotImplementedError(f"cannot fold {name}")
                self.val[res[0]] = ("const", out)
            else:
                raise NotImplementedError(f"MO op {name} on non-constant inputs")
            for i in ins:                       # free a value (a big weight!) after its last use
                self.uses[i] -= 1
                if self.uses[i] == 0 and i in self.val and i != self.out:
                    del self.val[i]


    # ---- constants
    def op_mo_constant(self, o, ins, res):
        shape, dt = _type(res[0])
        self.val[res[0]] = ("const", _dense(o, shape, dt))

    def op_mo_constant_external(self, o, ins, res):
        name = re.search(r'name = "([^"]+)"', str(o)).group(1)
        self.val[res[0]] = ("lazy", name)

    # ---- host ops: the embedding, the last-token select
    def op_rmo_mo_gather(self, o, ins, res):
        kind, idx = self.val[ins[1]]
        if kind == "tokens":                                 # embedding lookup, on the host (as hexagon_torch does)
            table = torch.from_numpy(np.ascontiguousarray(self.const(ins[0]), np.float32))
            tn = self.buffer("embed", table)
            e = self.fx.call_function(aten.embedding.default, (tn, idx))
            self.val[res[0]] = ("fx", self.f16(e))
        elif kind == "lastidx":                              # rows [input_row_offsets[1:] - 1]: batch 1, the last row
            x = self.node(ins[0])
            self.val[res[0]] = ("fx", self.fx.call_function(aten.slice.Tensor, (x, 0, -1, 9223372036854775807)))
        else:
            raise NotImplementedError(f"gather with {kind} indices")

    def op_rmo_slice(self, o, ins, res):
        if self.val[ins[0]][0] == "opaque":                   # input_row_offsets[1:]
            self.val[res[0]] = ("lastbase", None)
            return
        raise NotImplementedError("slice of a non-opaque value")

    def op_rmo_sub(self, o, ins, res):
        if self.val[ins[0]][0] == "lastbase":
            self.val[res[0]] = ("lastidx", None)
            return
        self.fold_default(o, ins, res)

    def op_mo_cast(self, o, ins, res):
        if self.is_const(ins[0]):
            return self.fold_default(o, ins, res)
        if _type(res[0])[1] != "f32":
            raise NotImplementedError("cast of a DSP value to a type other than f32")
        self.val[res[0]] = self.val[ins[0]]                  # the program returns fp32 anyway

    # ---- aliases
    def op_rmo_reshape(self, o, ins, res):
        if self.is_const(ins[0]):
            shape, dt = _type(res[0])
            self.val[res[0]] = ("const", self.const(ins[0]).reshape(shape))
            return
        kind, x = self.val[ins[0]]
        shape, _ = _type(res[0])
        if shape[0] != _type(ins[0])[0][0]:
            raise NotImplementedError("reshape that changes the row dimension")
        self.val[res[0]] = (kind, x)                  # [t, h, d] <-> [t, h*d]: same 2-D value on the DSP

    # ---- ops on the DSP
    def op_mo_reduce_rms_norm(self, o, ins, res):
        shape, _ = _type(ins[0])
        if len(shape) != 2:
            raise NotImplementedError("per-head rms_norm (Qwen3 QK-norm): S2 covers Llama3")
        gamma, eps = self.const(ins[1]), float(self.const(ins[2]))
        p = pack_rms_norm(gamma, eps)
        g = self.buffer("gamma", torch.from_numpy(p.gamma.copy()))
        self.val[res[0]] = ("fx", self.call(H.rms_norm.default, self.node(ins[0]), g, eps, p.k, LAYOUT_VERSION))

    def pack(self, w_out_in):
        """pack_linear_f16, through hexagon_torch's pack cache when one is open (lower.pack_cache():
        Generator opens one and fills it from worker processes before the programs are built)."""
        if lower._cache is None:
            return pack_linear_f16(w_out_in, None)
        key = lower.fingerprint(w_out_in, None, ("f16", "ref"))
        if key not in lower._cache:
            lower.cache_put(key, pack_linear_f16(w_out_in, None))
        return lower._cache[key]

    def linear(self, x, w_out_in):
        p = self.pack(w_out_in)
        wn = self.buffer("w", torch.from_numpy(p.weights.copy()))
        bn = self.buffer("bt", torch.from_numpy(p.bias_table.view(np.int32).copy()))
        return self.call(H.linear.default, x, wn, bn, p.fmt, p.k, p.n, LAYOUT_VERSION)

    def op_rmo_matmul(self, o, ins, res):
        w = self.const(ins[1])                         # [K, N] (the graph transposes [N, K])
        x = self.node(ins[0])
        self.val[res[0]] = ("matmul", (x, w))          # emitted when we know how it is consumed

    def materialize(self, v):
        kind, x = self.val[v]
        if kind == "matmul":
            xin, w = x
            node = self.linear(xin, np.ascontiguousarray(w.T))
            self.val[v] = ("fx", node)
            return node
        return self.node(v)

    def op_mo_split(self, o, ins, res):
        kind, (xin, w) = self.val[ins[0]]
        assert kind == "matmul", "split of a non-matmul"
        sizes = [_type(r)[0][-1] for r in res]
        off = 0
        for r, n in zip(res, sizes):
            self.val[r] = ("fx", self.linear(xin, np.ascontiguousarray(w[:, off:off + n].T)))
            off += n

    def op_rmo_mo_silu(self, o, ins, res):
        self.val[res[0]] = ("silu", self.materialize(ins[0]))

    def op_rmo_mul(self, o, ins, res):
        if all(self.is_const(i) for i in ins):
            return self.fold_default(o, ins, res)
        a, b = self.val[ins[0]], self.val[ins[1]]
        if a[0] == "silu":
            gate, up = a[1], self.materialize(ins[1])
        elif b[0] == "silu":
            gate, up = b[1], self.materialize(ins[0])
        else:
            raise NotImplementedError("mul that is not silu(gate) * up")
        self.val[res[0]] = ("fx", self.call(H.swiglu.default, gate, up, LAYOUT_VERSION))

    def fold_default(self, o, ins, res):
        shape, dt = _type(res[0])
        self.val[res[0]] = ("const", _fold(o.name, o, [self.const(i) for i in ins], shape, self.fdt(dt)))

    def op_rmo_add(self, o, ins, res):
        if all(self.is_const(i) for i in ins):
            return self.fold_default(o, ins, res)
        a, b = self.materialize(ins[0]), self.materialize(ins[1])
        self.val[res[0]] = ("fx", self.call(H.add.default, a, b, LAYOUT_VERSION))

    # ---- the custom ops
    def op_mo_custom(self, o, ins, res):
        sym = _symbol(o)
        if sym == "mo.rope_split_store.ragged.paged":
            return self.rope_split_store(o, ins, res)
        if sym == "mo.mha.ragged.paged":
            return self.mha(o, ins, res)
        raise NotImplementedError(f"custom op {sym}")

    def rope_split_store(self, o, ins, res):
        if _param(o, "interleaved") != "false":
            raise NotImplementedError("interleaved RoPE")
        qkv_v, freqs = ins[0], self.const(ins[2])
        kind, (xin, w) = self.val[qkv_v]
        assert kind == "matmul", "rope_split_store of a non-matmul"
        kv_shape = _type_buffer(ins[3])                 # [pages, 2, layers, page, n_kv, head_dim]
        n_kv, hd = kv_shape[-2], kv_shape[-1]
        n_q = _type(res[0])[0][-1] // hd
        self.n_kv, self.hd, self.n_heads = n_kv, hd, n_q
        # (cos, sin) interleaved; MAX builds the table for twice the context, the DSP op wants max_seq rows
        cos = np.ascontiguousarray(freqs[:self.max_seq, 0::2], np.float32)
        sin = np.ascontiguousarray(freqs[:self.max_seq, 1::2], np.float32)
        ct, st = self.buffer("cos", torch.from_numpy(cos)), self.buffer("sin", torch.from_numpy(sin))
        qn, kn = n_q * hd, n_kv * hd
        q = self.linear(xin, np.ascontiguousarray(w[:, :qn].T))
        k = self.linear(xin, np.ascontiguousarray(w[:, qn:qn + kn].T))
        v = self.linear(xin, np.ascontiguousarray(w[:, qn + kn:qn + 2 * kn].T))
        t = self.t
        q = self.call(H.rope.default, q, ct, st, self.pos, t, hd, LAYOUT_VERSION)
        k = self.call(H.rope.default, k, ct, st, self.pos, t, hd, LAYOUT_VERSION)
        ms, layer = self.max_seq, int(self.const(ins[8]))
        kc = self.buffer(f"kcache{layer}_", torch.from_numpy(layouts.pack_kcache(np.zeros((n_kv, ms, hd), np.float16))))
        vc = self.buffer(f"vcache{layer}_", torch.from_numpy(layouts.pack_vcache(np.zeros((n_kv, ms, hd), np.float16))))
        self.call(H.index_copy.default, kc, k, self.pos, ops.KCACHE, n_kv, hd, ms, LAYOUT_VERSION)
        self.call(H.index_copy.default, vc, v, self.pos, ops.VCACHE, n_kv, hd, ms, LAYOUT_VERSION)
        self.caches[layer] = (kc.target, vc.target)
        self.cache_nodes[layer] = (kc, vc)
        self.val[res[0]] = ("fx", q)

    def mha(self, o, ins, res):
        mask = _param(o, "mask_str")
        if mask != "causal":
            raise NotImplementedError(f"mask {mask}")
        scale = float(self.const(ins[8]))
        kc, vc = self.cache_nodes[int(self.const(ins[7]))]
        y = self.call(H.scaled_dot_product_attention.default, self.node(ins[0]), kc, vc, self.pos,
                      self.n_heads, self.n_kv, self.hd, self.max_seq, True, scale, LAYOUT_VERSION)
        self.val[res[0]] = ("fx", y)

    # ---- result
    def finish(self):
        y = self.materialize(self.out)
        if self.batched:
            y = self.fx.call_function(aten.reshape.default, (y, [1, self.t, _type(self.out)[0][-1]]))
        self.fx.output(self.fx.call_function(aten._to_copy.default, (y,), {"dtype": torch.float32}))
        root = torch.nn.Module()
        for n, tns in self.gm_buffers.items():
            root.register_buffer(n, tns)
        gm = torch.fx.GraphModule(root, self.fx)
        return gm, self.counts


def _type_buffer(v):
    m = re.match(r"!mo\.buffer<\[(.*?)\], (\w+)>", str(v.type))
    return [int(d) if d.strip().isdigit() else d.strip() for d in m.group(1).split(",")]


def convert(graph, weights, t=None, pos=None, max_seq=256, batched=False, exact_tables=False):
    """MAX Graph (one block: input [t, hidden], kv inputs) -> (GraphModule of hexagon ops, op counts).
    The module takes x [t, hidden] fp32 and returns [t, hidden] fp32; its packed KV caches are the
    buffers named in `gm.caches`."""
    c = Converter(graph, weights, t, pos, max_seq, batched, exact_tables)
    gm, counts = c.run()
    gm.caches = c.caches if len(c.caches) != 1 else {"k": c.caches[0][0], "v": c.caches[0][1]}
    gm.layer_caches = c.caches
    return gm, counts
