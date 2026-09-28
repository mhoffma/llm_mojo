"""Prints checksums of dequantized rows of every matrix in a GGUF file, for
tests/gguf_check.py to compare with the Python `gguf` package.

    uv run mojo run -I . tests/gguf_dump.mojo gpt2/gguf/gpt2.Q4_K_M.gguf
"""

from std.sys import argv
from std.memory.alloc import unsafe_alloc

from gguf import GGUFFile, type_name


def main() raises:
    var g = GGUFFile(String(argv()[1]))
    var names = List[String]()
    for e in g.tensors.items():
        if e.value.rows > 1:
            names.append(e.key)
    sort(names)
    for name in names:
        var m = g.matrix(name)
        var buf = unsafe_alloc[Float32](m.cols)
        for row in [0, m.rows // 2, m.rows - 1]:
            m.dequant_row(row, buf)
            var s = Float64(0)
            var sq = Float64(0)
            for i in range(m.cols):
                var v = Float64(buf[unsafe_offset=i])
                s += v
                sq += v * v
            print(name, type_name(m.kind), row, s, sq, buf[unsafe_offset=0], buf[unsafe_offset=m.cols - 1])
        buf.unsafe_free()
