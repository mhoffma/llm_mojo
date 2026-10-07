"""Export a Llama-family checkpoint to an .hxb container with the programs made from MAX graphs.

Same command line as `python -m hexagon_torch.export_blob`, same container; the only change is that
Generator builds its layer-group and head programs through max_lower (gen_max.MaxGenerator). With
--stock the stock Generator is used instead, so both containers come from one script.

  HVXHMX_REPO=<repo> python export_mojomax.py --model <checkpoint dir> --out x.hxb \\
      --remote tcp://10.168.168.32:9872 --lib-dir /home/mhoffman/hexagon/mojomax/lib --skel-hash <sha256>
"""
import argparse
import os

import torch

import max_lower  # noqa: F401  (puts HVXHMX_REPO on sys.path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--layers", type=int, default=22)
    p.add_argument("--out", required=True)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--prefill-rows", type=int, default=32)
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--remote", default=None)
    p.add_argument("--lib-dir", default=None)
    p.add_argument("--skel-hash", default=None)
    p.add_argument("--weights", default="f16", choices=["f16", "w4f16", "w4f16v2"])
    p.add_argument("--head-weights", default="f16", choices=["f16", "w4f16", "w4f16v2"])
    p.add_argument("--keep-f16", default="", help="linears kept fp16 under a W4 --weights: kinds (k), layers (L2), L2.down")
    p.add_argument("--rounding", default="ref", choices=["ref", "search"])
    p.add_argument("--max-tables", action="store_true", help="keep MAX's own fp32 RoPE tables (default: the reference's fp64 ones)")
    p.add_argument("--stock", action="store_true", help="use hexagon_torch's own torch.export front end")
    a = p.parse_args()
    from hexagon_torch import export_blob, generate
    if not a.stock:
        import gen_max
        gen_max.MaxGenerator.checkpoint = os.path.abspath(a.model)
        gen_max.MaxGenerator.exact_tables = not a.max_tables
        generate.Generator = gen_max.MaxGenerator              # export_blob._export imports it at call time
    weights = a.weights
    if a.keep_f16 or a.rounding != "ref":
        from hexagon_torch.lower import WeightPolicy
        weights = WeightPolicy(a.weights, [k for k in a.keep_f16.split(",") if k], a.rounding)
    export_blob.export_model(a.model, a.layers, a.out, context=a.context, weights=weights, skel_hash=a.skel_hash,
                             head_weights=a.head_weights, prefill=not a.no_prefill, prefill_rows=a.prefill_rows,
                             dtype=torch.bfloat16, remote=a.remote, lib_dir=a.lib_dir)


if __name__ == "__main__":
    main()
