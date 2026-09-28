"""Checks gguf.mojo's dequantization against the Python `gguf` package.

    uv run --with gguf python tests/gguf_check.py gpt2/gguf/gpt2.Q4_K_M.gguf

Runs tests/gguf_dump.mojo on the file, dequantizes the same rows in Python,
and compares row sums, sums of squares, and the first and last values.
"""

import subprocess
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

path = sys.argv[1]
out = subprocess.run(["uv", "run", "mojo", "run", "-I", ".", "tests/gguf_dump.mojo", path],
                     capture_output=True, text=True, check=True).stdout
tensors = {t.name: t for t in GGUFReader(path).tensors}
worst, lines = 0.0, 0
for line in out.splitlines():
    name, kind, row, s, sq, first, last = line.split()
    t = tensors[name]
    w = dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0])).astype(np.float64)
    r = w[int(row)]
    want = [r.sum(), (r * r).sum(), r[0], r[-1]]
    got = [float(s), float(sq), float(first), float(last)]
    scale = max(1e-3, np.abs(r).max() * len(r) ** 0.5)
    err = max(abs(a - b) for a, b in zip(got, want[:1])) / scale
    err = max(err, abs(got[1] - want[1]) / max(1e-6, want[1]))
    err = max(err, max(abs(got[i] - want[i]) for i in (2, 3)) / max(1e-6, np.abs(r).max()))
    worst = max(worst, err)
    lines += 1
    if err > 1e-5:
        print("MISMATCH", name, kind, row, "got", got, "want", want)
print(f"{lines} rows checked, worst relative error {worst:.2e}")
sys.exit(0 if worst <= 1e-5 else 1)
