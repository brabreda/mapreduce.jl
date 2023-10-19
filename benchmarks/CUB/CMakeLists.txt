cmake_minimum_required(VERSION 3.20)
project(CUB_REDUCTION_BENCHMARK)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2/")

find_package(CUDA REQUIRED)


set(CUDA_FILES
  src/main.cu
)

set(CPP_FILES
  src/CUB_ReductionSample.cpp
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
CUDA_ADD_EXECUTABLE(CUB_Benchmark ${CUDA_FILES})


include_directories(src/)