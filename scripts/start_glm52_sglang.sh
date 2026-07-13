#!/usr/bin/env bash
# GLM-5.2-FP8 via sglang + ATOM plugin
# Base image: rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260621  (+ ATOM fork)
#
# Barrier C5 fix baked in: use bf16 / auto KV cache. Do NOT pass
# --kv-cache-dtype fp8_e4m3 — aiter's asm MLA decode has no gqa8 kernel for
# fp8 KV (get_heuristic_kernel_mla abort); bf16 KV loads mla_a16w16_...gqaratio8.
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
# (optional) non-preshuffle GEMM/quant if you hit preshuffle kernel gaps:
# export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0

python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --trust-remote-code --tp-size 8 \
  --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 16 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --context-length 32768 \
  --disable-cuda-graph --disable-custom-all-reduce
  # ^ NOTE: bf16/auto KV (no --kv-cache-dtype fp8_e4m3) — that's the C5 fix
