"""Unit test for the integer (A16) matmul kernels.

Compares kernels.matmul_rows_a16 against an exact float64 computation of the
same quantized arithmetic: activations quantized to int16 exactly as
quantize_rows does, weights dequantized with dequant_row. The integer sums
are exact, so the kernels should agree to float32 rounding (~1e-6 relative).

Covers the one-token path (dot_row_i16), the tile path (tile_i16, for T > 1),
the leftover rows when OUT isn't a multiple of 32 (rows_i16), token counts
that aren't a multiple of the 4-token tile, and inputs that would overflow
int32 without the tile path's flushing (large activations, int8 weights,
3072-wide groups).

    uv run mojo run -I . tests/a16_kernels.mojo
"""

from std.memory.alloc import unsafe_alloc
from std.random import random_float64, seed
from std.math import round

from tensor import FPtr, WeightMatrix, QuantMatrix
from kernels import matmul_rows_a16


def check[W: WeightMatrix](name: String, T: Int, IN: Int, OUT: Int, big: Bool) -> Float64:
    """Returns the largest error relative to the output's largest value."""
    var src = unsafe_alloc[Float32](IN * OUT)  # Hugging Face layout [IN, OUT]
    for i in range(IN * OUT):
        src[unsafe_offset=i] = Float32(random_float64(-0.1, 0.1))
    var w = W.from_f32(src, IN, OUT, reduce_rows=True)
    var x = unsafe_alloc[Float32](T * IN)
    for i in range(T * IN):
        # With `big`, every activation is near the row's max, so the int16
        # codes are all near +-32767: the worst case for int32 overflow.
        var v = random_float64(0.9, 1.0) if big else random_float64(-1, 1)
        x[unsafe_offset=i] = Float32(v * Float64(1 if i % 2 == 0 else -1) if big else v)
    var b = unsafe_alloc[Float32](OUT)
    for o in range(OUT):
        b[unsafe_offset=o] = Float32(random_float64(-1, 1))
    var out = unsafe_alloc[Float32](T * OUT)

    if T == 1:
        matmul_rows_a16(out, x, w, b, 1, IN, OUT)
    else:
        matmul_rows_a16(out, x, w, b, T, IN, OUT)

    # Exact reference: quantize x the same way, dequantize w, sum in float64.
    var row = unsafe_alloc[Float32](IN)
    var worst = Float64(0)
    var biggest = Float64(0)
    for t in range(T):
        var mx = Float32(0)
        for i in range(IN):
            mx = max(mx, abs(x[unsafe_offset = t * IN + i]))
        var s = mx / 32767
        var inv = 1 / s
        for o in range(OUT):
            w.dequant_row(o, row)
            var want = Float64(0)
            for i in range(IN):
                var q = round(x[unsafe_offset = t * IN + i] * inv)
                want += Float64(q) * Float64(row[unsafe_offset=i])
            want = want * Float64(s) + Float64(b[unsafe_offset=o])
            worst = max(worst, abs(Float64(out[unsafe_offset = t * OUT + o]) - want))
            biggest = max(biggest, abs(want))
    var rel = worst / biggest
    print(
        "  ", name, "T =", T, "IN =", IN, "OUT =", OUT, "big" if big else "",
        ": max error / max |out| =", rel, "OK" if rel < 1e-5 else "FAIL",
    )
    w.free()
    src.unsafe_free()
    x.unsafe_free()
    b.unsafe_free()
    out.unsafe_free()
    row.unsafe_free()
    return rel


def main():
    seed(7)
    var worst = Float64(0)
    print("A16 kernels vs exact float64:")
    for T in [1, 3, 4, 9]:
        worst = max(worst, check[QuantMatrix[4, 32, False, True]]("int4-g32", T, 768, 64, False))
        worst = max(worst, check[QuantMatrix[4, 32, False, True]]("int4-g32", T, 768, 70, False))
    for T in [1, 6]:
        worst = max(worst, check[QuantMatrix[8, 0, False, True]]("int8-ch", T, 768, 64, False))
        worst = max(worst, check[QuantMatrix[8, 0, False, True]]("int8-ch", T, 3072, 64, True))
        worst = max(worst, check[QuantMatrix[4, 0, True, True]]("int4-ch-sym", T, 3072, 40, True))
    print("worst:", worst, "PASS" if worst < 1e-5 else "FAIL")
