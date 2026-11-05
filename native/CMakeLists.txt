cmake_minimum_required(VERSION 3.18)
project(nps_raster LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Set -DCMAKE_CUDA_ARCHITECTURES to your GPU SM number, e.g. 89 for RTX 4060")
endif()
add_library(nps_raster SHARED bridge.cu
  ../vendor/rasterizer/cuda_rasterizer/forward.cu
  ../vendor/rasterizer/cuda_rasterizer/backward.cu
  ../vendor/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
target_include_directories(nps_raster PRIVATE ../vendor/rasterizer ../vendor/rasterizer/third_party/glm)
set_target_properties(nps_raster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(nps_raster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# No --use_fast_math: first establish numerical alignment.

