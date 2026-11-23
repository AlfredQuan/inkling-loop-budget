#!/usr/bin/env bash
set -euo pipefail

# MODEL/BIN 使用绝对路径。BIN 可指向 llama-app 或 llama-server。
depth=$1
work=$2
[[ "$depth" =~ ^[1-5]$ ]]
: "${MODEL:?请设置模型绝对路径}" "${BIN:?请设置后端绝对路径}"
python_bin=${PYTHON_BIN:-python3}
port=${PORT:-8397}
workers=${WORKERS:-4}
batch=$((workers * 256))
ctx=$((workers * 2048))
mkdir -p "$work"
test -s "$work/cases.jsonl"
cmd=("$BIN")
[[ "$(basename "$BIN")" == llama ]] && cmd+=(serve)
params=(-m "$MODEL" --host 127.0.0.1 --port "$port" -c "$ctx" -np "$workers" -b "$batch" -ub "$batch" --threads-http "$((workers + 2))"
  -ngl "${GPU_LAYERS:-all}" -t "${CPU_THREADS:-8}" --no-cache-prompt
  --reasoning-effort none --override-kv "inkling2_loop.loop.count=int:$depth" -lv 4)
if [[ "${GPU_LAYERS:-all}" == 0 ]]; then
  params+=(--device none --no-op-offload)
fi
printf '%q ' "${cmd[@]}" > "$work/depth-$depth.command"
printf '%q ' "${params[@]}" >> "$work/depth-$depth.command"
printf '\n' >> "$work/depth-$depth.command"
"${cmd[@]}" "${params[@]}" > "$work/depth-$depth.server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0; i<150; i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    tail -45 "$work/depth-$depth.server.log" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
  sleep 1
done
((ready == 1)) || { tail -30 "$work/depth-$depth.server.log" >&2; exit 1; }
grep -Eq "metadata override.*inkling2_loop\.loop\.count.*= ([0-9.]+ I )?$depth([[:space:]]|$)" "$work/depth-$depth.server.log" \
  || { grep -i 'loop.count' "$work/depth-$depth.server.log" >&2 || true; exit 1; }
grep -Eq "n_layer_all[[:space:]]*=[[:space:]]*$((12 + 12 * depth))([[:space:]]|$)" "$work/depth-$depth.server.log" \
  || { grep -i 'n_layer_all' "$work/depth-$depth.server.log" >&2 || true; exit 1; }
"$python_bin" "$(dirname "$0")/sweep.py" "$work/cases.jsonl" "$work/depth-$depth.jsonl" \
  --depth "$depth" --url "http://127.0.0.1:$port" --workers "$workers"
