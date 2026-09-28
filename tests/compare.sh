#!/usr/bin/env bash
# Runs the accuracy harness (--ppl --compare against float32) for several
# configurations and prints one table row per configuration.
#
# Usage: tests/compare.sh "OPTIONS" ["OPTIONS" ...]
#   Each argument is one configuration: the gpt2t_bin options to use.
#   Example: tests/compare.sh "--dtype f32 --kv int8" "--dtype int4-g32-a16"
#
# Each run takes ~45 s. Prints markdown: configuration, model name,
# perplexity and its change vs float32, top-1 agreement, mean KL, and the
# mean absolute logit difference.
set -u
cd "$(dirname "$0")/.."
export MODULAR_THREAD_BUSY_WAIT_US=0
[ $# -gt 0 ] || { sed -n '2,11p' "$0"; exit 1; }

echo "| Options | Model | Perplexity | vs f32 | Top-1 vs f32 | Mean KL | Mean logit diff |"
echo "|---|---|---|---|---|---|---|"
for opts in "$@"; do
  # shellcheck disable=SC2086
  out=$(./gpt2t_bin $opts --ppl tests/data/alice_ch1.txt --compare 2>&1)
  name=$(grep " perplexity " <<<"$out" | head -1 | awk '{print $1}')
  ppl=$(grep " perplexity " <<<"$out" | head -1 | awk '{printf "%.3f", $3}')
  chg=$(grep "change" <<<"$out" | sed -E 's/.*change (-?[0-9.e+-]+) %.*/\1/' | awk '{printf "%+.2f%%", $1}')
  top1=$(grep "top-1 agreement" <<<"$out" | awk '{printf "%.1f%%", $(NF-1)}')
  kl=$(grep "mean KL" <<<"$out" | awk '{printf "%.2e", $(NF-1)}')
  diff=$(grep "logit |diff|" <<<"$out" | sed -E 's/.*mean ([0-9.e+-]+) .*/\1/' | awk '{printf "%.3f", $1}')
  echo "| \`$opts\` | $name | $ppl | $chg | $top1 | $kl | $diff |"
done
