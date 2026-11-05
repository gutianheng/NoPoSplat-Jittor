#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-re10k}"
case "$dataset" in re10k|acid) ;; *) echo "Expected re10k or acid" >&2; exit 2;; esac
exec python tools/evaluate_resumable.py --config "configs/extended_20261103/official_$dataset.yaml" \
  --weights "${NPS_WEIGHTS:-weights/$dataset.npz}" \
  --lpips-weights "${NPS_LPIPS_WEIGHTS:-weights/lpips_vgg.npz}" \
  --weights-role official_pretrained --output "${NPS_OUTPUT:-runs/official_${dataset}_jittor}"
