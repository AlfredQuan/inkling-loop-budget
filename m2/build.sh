#!/usr/bin/env bash
set -euo pipefail
: "${LLAMA_SRC:?Set LLAMA_SRC to the pinned llama.cpp source directory}"
: "${LLAMA_LIB:?Set LLAMA_LIB to the directory containing libllama and libggml}"
out="${1:?Usage: build.sh output-binary}"
g++ -std=c++17 -O2 -Wall -Wextra -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
    "$(dirname "$0")/probe.cpp" -L"$LLAMA_LIB" -Wl,-rpath,"$LLAMA_LIB" \
    -lllama -lggml -lggml-base -o "$out"
