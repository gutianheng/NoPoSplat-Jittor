#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
framework="${1:-jittor}"; variant="${2:-default}"
case "$framework" in jittor|torch) ;; *) echo "Expected jittor or torch" >&2; exit 2;; esac
case "$variant" in default|no_intrinsics|no_rgb) ;; *) echo "Unknown variant" >&2; exit 2;; esac
init="reference/mast3r_init/encoder.npz"
if [[ "$variant" == "no_rgb" ]]; then init="reference/mast3r_no_rgb_init/encoder.npz"; fi
exec python tools/reduced_training.py "$framework" --config "configs/extended_20261103/$variant.yaml" \
  --official "${NPS_OFFICIAL_DIR:-../NoPoSplat-original}" \
  --weights "${NPS_INIT_WEIGHTS:-$init}" \
  --lpips-weights "${NPS_LPIPS_WEIGHTS:-weights/lpips_vgg.npz}" \
  --stop-step 3000 --output "${NPS_OUTPUT:-runs/${variant}_${framework}}"
