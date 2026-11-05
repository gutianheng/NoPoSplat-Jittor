# Environment, data, training and evaluation

## 1. Environments

The Jittor runtime and PyTorch reference/export environment are independent. Do not install a Torch model in the Jittor training path as a substitute backend.

Tested early GPU: RTX4090D, CUDA11.8, SM89. Final extended experiments: RTXPRO6000 Blackwell96GB, CUDA12.8, Jittor1.3.11.0; referenceTorch2.7.1+cu128. Full precision FP32; noAMP/TF32.

On Linux, create Python3.10 venv, install `-e .`, set `NPS_CUDA_ARCH` to the actual GPU architecture and run `scripts/build_rasterizer.sh`. C++ compiler, CMake and nvcc are required. See `scripts/server_preflight.sh` and `tools/check_native.py`.

**Blackwell caution:** unmodified Jittor1.3.11.0 and CUDA11.8 are not a tested SM120 stack. `tools/platform` preserves architecture/cuDNN compatibility patch sources from the experiment. These scripts use an isolated `/root/nps-blackwell` prefix and assert the expected installed-source text. They are diagnostic, version-specific patches, not a universal installer. In addition to SM120 selection, cuDNN9 consolidated-library discovery must be adapted; the generic quick start is for the earlier Ada toolchain. A ready-to-use Blackwell container is not claimed to be published. Never reuse a compiled Ada cache on SM120.

## 2. Prepare original checkout, data and weights (reference environment)

Follow the official NoPoSplat environment and dataset instructions and obtain trusted official checkpoints. Clone the reference outside this repository, e.g. `../NoPoSplat-original`.

```bash
# Torch reference environment
python tools/convert_checkpoint.py --input /data/re10k.ckpt --output weights/re10k.npz --trusted-input
python tools/export_lpips.py --output weights/lpips_vgg.npz
python tools/convert_dataset.py --source /data/re10k --output data/extended_20261103/re10k --stage test --evaluation-index configs/extended_20261103/evaluation_index_re10k.json --trusted-input
```

For ACID use `weights/acid.npz`, output root `data/extended_20261103/acid` and the frozen ACID index. Prepare the exact training candidates and separate initialisation for the no-RGB-head variant:

```bash
python tools/convert_dataset.py --source /data/re10k --output data/extended_20261103/re10k --stage train --evaluation-index configs/extended_20261103/training_candidates.json --trusted-input
python tools/export_reference.py --official ../NoPoSplat-original --mast3r /data/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --output reference/mast3r_init --cpu --seed 111123 --trusted-input
python tools/export_reference.py --official ../NoPoSplat-original --mast3r /data/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --output reference/mast3r_no_rgb_init --cpu --seed 111123 --override model.encoder.gs_params_head_type=dpt --trusted-input
```

The no-intrinsics run uses the default initialisation; the no-RGB-head run has a different parameter schema and needs the matching no-RGB export. An official trained NoPoSplat checkpoint is not a substitute for MASt3R training initialisation. Converting arbitrary train scenes instead of the provided3000 candidates changes the experiment. The mirror-plan files additionally record the frozen source revisions/chunkSHA values used in the experiment.

Use only trusted Torch serialisation files; do not apply `--trusted-input` to untrusted downloads. The converted NPZ readers do not require Torch in the training process.

## 3. Training (separate environments)

The extended configuration retains a5000-update schedule but stops at3000 to match the executed protocol. Edit dataset root paths to your converted data; preserve image, camera, optimiser and schedule settings.

```bash
# Jittor environment
python tools/reduced_training.py jittor --config configs/extended_20261103/default.yaml --weights reference/mast3r_init/encoder.npz --lpips-weights weights/lpips_vgg.npz --stop-step 3000 --output runs/default_jittor
# independent Torch reference environment
python tools/reduced_training.py torch --official ../NoPoSplat-original --config configs/extended_20261103/default.yaml --weights reference/mast3r_init/encoder.npz --lpips-weights weights/lpips_vgg.npz --stop-step 3000 --output runs/default_torch
```

Repeat with `no_intrinsics.yaml` and `no_rgb.yaml`. Each variant must train independently at the same budget; switching modules off only during evaluation is not this training ablation. Default shape: full model, not tiny. Respect actual memory requirements; batch1 on24GB and batch4 on96GB were measured, not a guarantee for every platform/configuration.

## 4. Evaluation

```bash
# Jittor environment; official weights
python tools/evaluate_resumable.py --config configs/extended_20261103/official_re10k.yaml --weights weights/re10k.npz --lpips-weights weights/lpips_vgg.npz --weights-role official_pretrained --output runs/official_re10k_jittor
# resume the SAME protocol and weights
python tools/evaluate_resumable.py --config configs/extended_20261103/official_re10k.yaml --weights weights/re10k.npz --lpips-weights weights/lpips_vgg.npz --weights-role official_pretrained --resume-evaluation --output runs/official_re10k_jittor
```

Configuration filenames present in `configs/extended_20261103` are authoritative; use `official_re10k.yaml` or `official_acid.yaml`. For own weights select the corresponding trained `encoder_003000.npz` and `--weights-role own_reduced_training`. The Torch evaluator is `tools/evaluate_torch_resumable.py`; inspect its `--help` for the independent reference-checkout argument.

Per-scene atomic journals bind weights, fixed index, preprocessing and pose settings. A scene is committed only after all its targets complete. Do not silently skip missing scenes. The CLI has a separate no-alignment option; report that as a different protocol.

## 5. Results and tests

Six measured training CSVs and ten evaluation summaries/CSVs are included in `results`. `docs/executed_core_sha256.json` verifies the published neural mathematical core and native bridge against the executed snapshot. CPU tests run independently of GPU qualification; strict failures remain in the limitations document.

## Compact presentation entry points

After the preparation above:

```bash
bash scripts/train_reduced.sh jittor default
bash scripts/train_reduced.sh jittor no_intrinsics
bash scripts/train_reduced.sh jittor no_rgb
bash scripts/test_official.sh re10k
bash scripts/test_official.sh acid
```

Set `NPS_INIT_WEIGHTS`, `NPS_WEIGHTS`, `NPS_LPIPS_WEIGHTS`, `NPS_OFFICIAL_DIR` and `NPS_OUTPUT` when using other local paths. Do not reuse one output directory for different protocols.
