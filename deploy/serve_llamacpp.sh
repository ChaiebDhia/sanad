#!/usr/bin/env sh
# CPU / edge serving path. UNVERIFIED in this repo's CI.
# 1) Convert the merged model to GGUF with llama.cpp's convert script, 2) quantise (e.g. Q4_K_M),
# 3) serve it. Then point `sanad eval/bench --base-url http://localhost:8080/v1` at it.
# Usage: deploy/serve_llamacpp.sh model-q4_k_m.gguf
set -eu
exec llama-server -m "${1:?path to .gguf}" --port 8080
