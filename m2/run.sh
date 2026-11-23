#!/usr/bin/env bash
set -euo pipefail

: "${PROBE:?Set PROBE to the binary produced by m2/build.sh}"
: "${MODEL:?Set MODEL to the pinned IQ4_NL GGUF}"
work="${1:?Usage: run.sh work-dir-with-manifest-and-prompts}"
dir=$(cd "$(dirname "$0")" && pwd)
py="${PYTHON_BIN:-python3}"
test -s "$work/manifest.jsonl"
test -s "$work/prompts.tsv"
test "$(sha256sum "$MODEL" | cut -d ' ' -f 1)" = e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2
exec 9>"$work/sweep.lock"
flock -n 9

for d in 1 2 3; do
    f="$work/depth-$d.jsonl"
    if [[ -e "$f" ]]; then
        printf 'Use a fresh output directory; %s already exists.\n' "$f" >&2
        exit 1
    fi
done

rows=$(wc -l < "$work/manifest.jsonl")
for d in 1 2 3; do
    f="$work/depth-$d.jsonl"
    "$PROBE" "$MODEL" "$d" 4 "$work/prompts.tsv" "$f.part" > "$work/depth-$d.log" 2>&1 \
        || { tail -40 "$work/depth-$d.log" >&2; exit 1; }
    test "$(wc -l < "$f.part")" -eq "$rows"
    grep -F "n_layer_all           = $((12 + 12*d))" "$work/depth-$d.log"
    grep -F "offloaded $((13 + 12*d))/$((13 + 12*d)) layers to GPU" "$work/depth-$d.log"
    mv "$f.part" "$f"
done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" "$dir/fit.py" \
    "$work/paired.jsonl.gz" "$work/results" --join "$work"
