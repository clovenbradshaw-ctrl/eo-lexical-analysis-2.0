#!/usr/bin/env bash
# Fetch the two local GGUF models (idempotent — skips files already present).
set -euo pipefail
cd "$(dirname "$0")/weights"

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "skip $out (already present)"
    return
  fi
  echo "fetching $out"
  curl -sS -L -o "$out" "$url"
}

fetch "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
      "qwen2.5-7b-instruct-q4_k_m.gguf"

fetch "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf" \
      "mistral-7b-instruct-v0.3-q4_k_m.gguf"

ls -la .
