#!/usr/bin/env sh
# GPU serving path. UNVERIFIED in this repo's CI: check flags against your vLLM version.
# Usage: deploy/serve_vllm.sh runs/sft-merged sanad-1.5b
set -eu
MODEL="${1:?path or HF id of the merged model}"
NAME="${2:-sanad}"
exec python -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$NAME" --port 8000
