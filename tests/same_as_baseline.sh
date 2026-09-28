#!/usr/bin/env bash
# Checks that gpt2t (generic formats) with --dtype f32 gives exactly the same
# output as the baseline gpt2: the same prompt tokens, the same top-5 logits
# printed to full float32 precision, and the same generated text, greedy and
# sampled, for a short and a long prompt.
#
# Usage: tests/same_as_baseline.sh    (from ~/fun, after building both:
#          uv run mojo build -o gpt2_bin gpt2.mojo
#          uv run mojo build -o gpt2t_bin gpt2t.mojo)
set -u
cd "$(dirname "$0")/.."
export MODULAR_THREAD_BUSY_WAIT_US=0

long=$(python3 -c "print(' '.join(['The history of computing is a long and winding story of people, machines and ideas.']*28))")
fail=0
check() {  # check <description> <args...>
    local desc=$1; shift
    local a b
    a=$(./gpt2_bin "$@" 2>/dev/null)
    b=$(./gpt2t_bin --dtype f32 "$@" 2>/dev/null)
    if [ "$a" == "$b" ]; then
        echo "SAME  $desc"
    else
        echo "DIFF  $desc"
        diff <(echo "$a") <(echo "$b") | head -10
        fail=1
    fi
}
check "short prompt, logits + greedy"  -v -t 0 -n 40 "Hello, my name is"
check "short prompt, sampled"          -v -n 40 -s 7 "Hello, my name is"
check "long prompt, logits + greedy"   -v -t 0 -n 40 "$long"
check "long prompt, sampled"           -v -n 40 -s 7 "$long"
check "empty prompt, sampled"          -n 40 -s 3 ""
exit $fail
