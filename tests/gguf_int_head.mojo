"""Unit test for the integer GGUF output heads (Q6_K and Q8_0, gguf.mojo).

kernels.head computes a GGUF head stored as Q6_K or Q8_0 in integers:
activations quantized to int16 (quantize_rows), weights as int16, sums in
int32 with VPDPWSSD. For each real head, this checks that

  1. one token (GGUFMatrix.dot_row_i16, what decode uses) and several tokens
     (unpack_row_i16 + dot4_i16 / dot1_i16, what evaluation uses) give
     identical logits, so decode and perplexity see the same model;
  2. the logits agree with an exact float64 computation of the same
     quantized arithmetic (weights from dequant_row, activations from their
     int16 codes) to float32 rounding;
  3. the same holds with every activation near +-32767 (large int32 sums).

Run: uv run mojo run -I . tests/gguf_int_head.mojo
     (needs the GGUF files from PLAN.md > Setup in gpt2/gguf/)
"""

from std.memory.alloc import unsafe_alloc
from std.random import random_float64, seed

from gguf import GGUFFile, GGUFMatrix, type_name
from kernels import head, quantize_rows


def check(path: String, big: Bool) raises -> Float64:
    """Returns the largest error relative to the largest |logit|, or 1 if
    the one-token and several-token paths differ."""
    var g = GGUFFile(path)
    var w = g.matrix("output.weight")
    var V = w.rows
    var C = w.cols
    comptime T = 5  # one 4-token block (dot4_i16) and one leftover (dot1_i16)
    var h = unsafe_alloc[Float32](T * C)
    for i in range(T * C):
        var v = random_float64(0.9, 1.0) * Float64(1 if i % 2 == 0 else -1) if big else random_float64(-3, 3)
        h[unsafe_offset=i] = Float32(v)
    var many = unsafe_alloc[Float32](T * V)
    head(many, h, w, T, V, C)
    var one = unsafe_alloc[Float32](V)

    # Exact reference: the int16 activations times the dequantized weights.
    var xq = unsafe_alloc[Int16](T * C)
    var sx = unsafe_alloc[Float32](T)
    quantize_rows(h, T, C, xq, sx)
    var row = unsafe_alloc[Float32](C)
    var worst = Float64(0)
    var mx = Float64(0)
    var differ = 0
    for t in range(T):
        head(one, h.unsafe_offset(t * C), w, 1, V, C)
        for v in range(V):
            if one[unsafe_offset=v] != many[unsafe_offset = t * V + v]:
                differ += 1
        for v in range(0, V, 7):  # every 7th row keeps the float64 loop short
            w.dequant_row(v, row)
            var d = Float64(0)
            for i in range(C):
                d += Float64(row[unsafe_offset=i]) * Float64(xq[unsafe_offset = t * C + i])
            d *= Float64(sx[unsafe_offset=t])
            mx = max(mx, abs(d))
            worst = max(worst, abs(Float64(many[unsafe_offset = t * V + v]) - d))
    var rel = worst / mx
    print(
        "  ", type_name(w.kind), "big" if big else "normal", ": one token vs",
        T, "tokens differ in", differ, "logits; max error / max |logit| =", rel,
    )
    h.unsafe_free()
    many.unsafe_free()
    one.unsafe_free()
    xq.unsafe_free()
    sx.unsafe_free()
    row.unsafe_free()
    # w points into g's buffer: keep g alive until here (Mojo destroys a
    # value right after its last use, here g.matrix).
    _ = g^
    return 1.0 if differ != 0 else rel


def main() raises:
    seed(1)
    var worst = Float64(0)
    for path in ["gpt2/gguf/gpt2.Q4_K_M.gguf", "gpt2/gguf/gpt2.Q8_0.gguf"]:
        for big in [False, True]:
            worst = max(worst, check(path, big))
    print("worst:", worst, "PASS" if worst < 1e-5 else "FAIL")
