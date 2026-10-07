"""A hexagon_torch Generator whose programs come from MAX.

`Generator._program(mod, rows, ...)` turns a module (the decoder layers [i0, i1), or the head) into a
lowered, planned program for one row count: torch.export + lower.lower. Here the same program is made
from a MAX graph of the same layers and converted by max_lower: everything after that (memory plan,
Program, the shared KV state, prefill/decode programs, export_blob) is hexagon_torch's own.
"""
import os

import numpy as np
import torch

import max_lower  # noqa: F401  (puts HVXHMX_REPO on sys.path)
from hexagon_torch import lower, memplan
from hexagon_torch.build import Program
from hexagon_torch.generate import Generator
from hexagon_torch.models import llama

import build_model


class MaxGenerator(Generator):
    checkpoint = None          # the checkpoint directory (set by export_mojomax.py before Generator is built)

    def _program(self, mod, rows, weights="f16", kv8=None):
        if kv8 is not None:
            raise NotImplementedError("int8 KV caches through the MAX front end")
        if weights not in ("f16",) and not (isinstance(weights, str) and weights == "f16"):
            raise NotImplementedError("only fp16 weights through the MAX front end so far")
        cfg = self.m.cfg
        if isinstance(mod, llama.Layers):
            graph, w = build_model.build_layers(self.checkpoint, mod.i0, mod.i1, max_seq=cfg.max_seq)
        elif isinstance(mod, llama.Head):
            if len(mod.slices):
                raise NotImplementedError("a vocabulary wider than one DSP tensor (head slices)")
            graph, w = build_model.build_head(self.checkpoint, max_seq=cfg.max_seq)
        else:
            raise TypeError(type(mod))
        gm, _ = max_lower.convert(graph, w, t=rows, pos=0, max_seq=cfg.max_seq, batched=True)
        with torch.no_grad():
            low = torch.export.export(gm, (torch.randn(1, rows, cfg.hidden),))
        return Program(self.s, low, memplan.plan_memory(low, self.s.vtcm_bytes, context=self.context), kv8=kv8)
