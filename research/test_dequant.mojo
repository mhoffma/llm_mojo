"""Probe: how fast does gguf.mojo dequantize each block format?

After SIMD dequantization, GGUF decoding ran at about the same speed for
Q8_0 (167 MB) and Q4_K_M (105 MB), so it is limited by arithmetic, not memory.
This measures, on one thread and per format:

  1. dequant_row alone: rows expanded to a float32 buffer (what matmul_rows
     does for prefill, before the tile kernel),
  2. dequant_row + a dot product with an activation vector, reading the
     buffer back (how decode first worked),
  3. dot_row: the dot product computed while dequantizing, with no buffer
     (how decode works now),

in billions of weights per second, over one real tensor of each format from
the GGUF files. The rows cycle through a whole matrix, so the weights come
from memory as they would when decoding.

Run: uv run mojo run -I . research/test_dequant.mojo
     (needs the GGUF files from PLAN.md > Setup in gpt2/gguf/)

Results on the i7-1160G7, one thread, G weights/s, 2026-09-27:

                      dequant only   dequant, then dot   fused dot_row
  Q8_0  ffn_up           8.6              7.0               17.2
  Q4_0  ffn_up          14.1              7.6               11.3
  Q4_1  ffn_up          18.6              8.6               14.5
  Q4_K  ffn_up          13.3              5.7                9.9
  Q5_K  attn_qkv        10.3              6.5               11.2
  Q6_K  output head      8.3              5.8                8.8

Conclusion: writing a dequantized row to memory and reading it back costs
more than the dequantization itself. Fusing the dot product into the
dequantizer is 1.3-2.5x faster, and it's how decode now works. Q6_K (the
output head, 38.6M weights per token) and Q4_K are the slowest per weight,
so they are where further decode speedups are.
"""

from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

from gguf import GGUFFile, GGUFMatrix, type_name
from tensor import F32V, NW


def bench(m: GGUFMatrix, reps: Int) -> Tuple[Float64, Float64, Float64]:
    var buf = unsafe_alloc[Float32](m.cols)
    var x = unsafe_alloc[Float32](m.cols)
    for i in range(m.cols):
        x[unsafe_offset=i] = Float32(i % 7) * 0.01
    var n = Float64(reps * m.rows * m.cols) / 1e9

    var t0 = perf_counter_ns()
    for _ in range(reps):
        for r in range(m.rows):
            m.dequant_row(r, buf)
    var only = n / (Float64(perf_counter_ns() - t0) / 1e9)

    var check = Float32(0)
    t0 = perf_counter_ns()
    for _ in range(reps):
        for r in range(m.rows):
            m.dequant_row(r, buf)
            var d = F32V(0)
            for i in range(0, m.cols, NW):
                d = buf.unsafe_load[width=NW](i).fma(x.unsafe_load[width=NW](i), d)
            check += d.reduce_add()
    var with_dot = n / (Float64(perf_counter_ns() - t0) / 1e9)

    t0 = perf_counter_ns()
    for _ in range(reps):
        for r in range(m.rows):
            check += m.dot_row(r, x)
    var fused = n / (Float64(perf_counter_ns() - t0) / 1e9)
    buf.unsafe_free()
    x.unsafe_free()
    if check == 12345:  # keep the result alive
        print(check)
    return (only, with_dot, fused)


def main() raises:
    # One large matrix of each format, from the files in PLAN.md's setup.
    var cases = [
        ("gpt2/gguf/gpt2.Q8_0.gguf", "blk.0.ffn_up.weight"),
        ("gpt2/gguf/gpt2.Q4_0.gguf", "blk.0.ffn_up.weight"),
        ("gpt2/gguf/gpt2.Q4_1.gguf", "blk.0.ffn_up.weight"),
        ("gpt2/gguf/gpt2.Q4_K_M.gguf", "blk.0.ffn_up.weight"),
        ("gpt2/gguf/gpt2.Q4_K_M.gguf", "blk.0.attn_qkv.weight"),
        ("gpt2/gguf/gpt2.Q4_K_M.gguf", "output.weight"),
    ]
    print("one thread, G weights/s:   dequant only | dequant, then dot | fused dot_row")
    for c in cases:
        var g = GGUFFile(c[0])
        var m = g.matrix(c[1])
        var reps = max(1, 20_000_000 // (m.rows * m.cols))
        _ = bench(m, 1)  # warm up
        var r = bench(m, reps)
        print(
            "  ", type_name(m.kind), c[1], "[", m.rows, "x", m.cols, "]:",
            r[0], "|", r[1], "|", r[2],
        )
        # m points into g's buffer: keep g alive until here (Mojo destroys a
        # value right after its last use, here g.matrix).
        _ = g^
