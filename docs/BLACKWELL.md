# Blackwell/SM120 compatibility notes

The measured extended experiment used CUDA12.8, cuDNN9 and Jittor1.3.11.0 in an isolated Python3.10 environment. A complete GPU image is not distributed. The following preserves the version-specific adaptations used in that environment; new installations still require the arithmetic/native/gradient checks.

## Prefix contract

Prepare a CUDA12.8 development toolkit with cuDNN9 headers and shared libraries. In the commands below, `$NPS_BLACKWELL_PREFIX/cuda-12.8` must point to that actual SDK and `$NPS_BLACKWELL_PREFIX/venv` to a Python3.10 Jittor1.3.11.0 venv. Do not point these variables to the host's unrelated old CUDA11.8 installation. Download/install NVIDIA components using their official distribution and licence terms.

```bash
export NPS_BLACKWELL_PREFIX=/root/nps-blackwell
export CUDA_HOME="$NPS_BLACKWELL_PREFIX/cuda-12.8"
export PATH="$CUDA_HOME/bin:$NPS_BLACKWELL_PREFIX/venv/bin:$PATH"
export nvcc_path="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NPS_CUDA_ARCH=120
export NVIDIA_TF32_OVERRIDE=0
export use_parallel_op_compiler=0
nvcc --list-gpu-code  # verify native sm_120 exists
```

## Apply once to the isolated installation

```bash
python tools/platform/patch_cudnn9_discovery.py
python tools/platform/patch_blackwell_arch.py
python tools/platform/patch_blackwell_cudnn_no_rnn.py
# Optional semantic guard for a TF32 comparison; final results used FP32.
python tools/platform/patch_blackwell_tf32_semantics.py
bash scripts/build_rasterizer.sh
```

The discovery patch selects cuDNN9 consolidated library names instead of guessing cuDNN8 names. The architecture patch raises Jittor's SM ceiling only when nvcc12.8 supports SM120. The RNN filter omits obsolete cuDNN RNN operators, which are not used by NoPoSplat; it is not a general Jittor RNN implementation. Backups and source hashes are preserved; repeated or mismatched patches fail rather than replacing arbitrary code. The patch sources derive from the executed experiment; the new portable-prefix wrappers were syntax checked, not freshly GPU-qualified after publication.

On the measured Ubuntu22.04 image an older Conda libstdc++ conflicted with generated C++ objects. The experiment preloaded the compatible system library `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`; inspect your own installation before applying that path. Jittor may also require a valid `python_config_path` and Python development headers. Never fix this by deleting libraries from another working environment.

Use a separate PyTorch2.7.1+cu128 reference environment on SM120; Torch2.1.2/cu118 is the earlier Ada reference, not the qualified Blackwell GPU stack. After setup, run the CUDA arithmetic/gradient probe, `tools/check_native.py` and real-model alignment tools before training. Compilation alone is not validation.
