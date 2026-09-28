"""Checks the Mojo tokenizer against OpenAI's tiktoken on tricky strings.

Usage (from ~/fun, after building gpt2t_bin):
    uv run --with tiktoken python tests/tokenizer_vs_tiktoken.py [binary]

`binary` defaults to ./gpt2t_bin; ./gpt2_bin works too. Exits non-zero on any
mismatch.
"""

import ast
import subprocess
import sys

import tiktoken

TESTS = [
    "Hello, my name is",
    "The quick brown fox jumps over the lazy dog. It's 2026, isn't it? We'll see; they'd've known.",
    "  leading spaces\n\nand   multiple    spaces\tand tabs\n",
    "numbers 1234567 and 3.14159, emails a.b@c.com, URLs https://x.org/a?b=c",
    "Unicode: café naïve résumé — em dash “quotes” … 日本語 Привет мир 😀!",
    "CAPS'S and O'Neil's 'quoted' ''double''",
    "trailing spaces   ",
    "def f(x):\n    return x**2  # comment\n",
]


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "./gpt2t_bin"
    enc = tiktoken.get_encoding("gpt2")
    bad = 0
    for s in TESTS:
        out = subprocess.run([binary, "-v", "-n", "0", s], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("prompt tokens:"))
        mine = ast.literal_eval(line.split(":", 1)[1].strip())
        ref = enc.encode(s)
        ok = mine == ref
        bad += not ok
        print("OK  " if ok else "DIFF", repr(s[:50]), len(ref))
        if not ok:
            print("   mine", mine)
            print("   ref ", ref)
    print("mismatches:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
