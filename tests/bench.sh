#!/usr/bin/env bash
# Benchmarks prompt (prefill) and per-token (decode) speed of weight formats,
# interleaving them round by round so drift in machine speed (heat, power
# source) affects every format alike.
#
# Usage: tests/bench.sh [ROUNDS] FORMAT...
#   FORMAT is a --dtype name (f32, f16, int4-g32, ...) or a GGUF file path.
#   Example: tests/bench.sh 3 f32 f16 gpt2/gguf/gpt2.Q4_K_M.gguf
#
# For each round and format: a 1-token prompt then 100 generated tokens
# ("short"), and a 476-token prompt then 100 generated tokens ("long").
# Top-k 1 makes the generated text, and so the work, the same every run.
# Prints one line per run, then the median of each column per format.
set -u
cd "$(dirname "$0")/.."
export MODULAR_THREAD_BUSY_WAIT_US=0

rounds=2
if [[ ${1:-} =~ ^[0-9]+$ ]]; then rounds=$1; shift; fi
[ $# -gt 0 ] || { sed -n '2,13p' "$0"; exit 1; }

ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null || echo "?")
[ "$ac" = 1 ] || echo "warning: not on AC power (AC/online=$ac); speeds will be low and noisy"

long=$(python3 -c "print(' '.join(['The history of computing is a long and winding story of people, machines and ideas.']*28))")
results=$(mktemp)
for r in $(seq "$rounds"); do
  for f in "$@"; do
    if [[ $f == *.gguf ]]; then args=(--gguf "$f"); name=$(basename "$f" .gguf); else args=(--dtype "$f"); name=$f; fi
    s=$(./gpt2t_bin "${args[@]}" -n 100 -k 1 "Hello" 2>&1 >/dev/null | tail -1)
    l=$(./gpt2t_bin "${args[@]}" -n 100 -k 1 "$long" 2>&1 >/dev/null | tail -1)
    # "... generated N tokens in X ms ( R tok/s )" -> R;  "prompt 476 tokens in X ms ( R tok/s )" -> R
    ds=$(sed -E 's/.*generated [0-9]+ tokens in [0-9]+ ms \( ([0-9]+) tok\/s \).*/\1/' <<<"$s")
    pl=$(sed -E 's/.*prompt [0-9]+ tokens in [0-9]+ ms \( ([0-9]+) tok\/s \).*/\1/' <<<"$l")
    dl=$(sed -E 's/.*generated [0-9]+ tokens in [0-9]+ ms \( ([0-9]+) tok\/s \).*/\1/' <<<"$l")
    mb=$(sed -E 's/.*\| ([0-9]+) MB \|.*/\1/' <<<"$l")
    printf "round %d  %-14s %4s MB  prompt %4s tok/s  decode short %4s  long %4s tok/s\n" "$r" "$name" "$mb" "$pl" "$ds" "$dl"
    echo "$name $mb $pl $ds $dl" >>"$results"
  done
done
echo
echo "median over $rounds rounds (tok/s):"
printf "%-14s %6s %8s %13s %12s\n" format MB prompt "decode short" "decode long"
python3 - "$results" <<'EOF'
import sys, statistics
rows = {}
for line in open(sys.argv[1]):
    name, *vals = line.split()
    rows.setdefault(name, []).append([int(v) for v in vals])
for name, runs in rows.items():
    med = [int(statistics.median(col)) for col in zip(*runs)]
    print(f"{name:14s} {med[0]:6d} {med[1]:8d} {med[2]:13d} {med[3]:12d}")
EOF
rm -f "$results"
