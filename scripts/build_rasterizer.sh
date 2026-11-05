#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NPS_CUDA_ARCH:?Set the GPU SM number, e.g. 80=A100, 86=RTX3090, 89=RTX4090/4060}"
test -f vendor/rasterizer/cuda_rasterizer/rasterizer.h
test -f vendor/rasterizer/third_party/glm/glm/glm.hpp
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES="$NPS_CUDA_ARCH"
cmake --build native/build --parallel "${NPS_BUILD_JOBS:-4}"
printf '%s\n' 'Built native/build/libnps_raster.so. Compilation is not a forward/gradient test.'

