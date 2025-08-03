//===- MTRTRuntime.cpp ----------------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of C runtime support library.
///
//===----------------------------------------------------------------------===//
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#ifdef MLIR_TRT_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#define HANDLE_CUDART_ERROR(x, ...)                                            \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      const char *msg = "";                                                    \
      msg = cudaGetErrorString(err);                                           \
      llvm::report_fatal_error(                                                \
          llvm::formatv("{0}#{1}: {2}\n", __FILE__, __LINE__, msg)             \
              .str()                                                           \
              .c_str());                                                       \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

/// Function which initializes the CUDA Runtime when the shared library is
/// loaded.
__attribute__((constructor)) static void initialize_cuda_runtime() {
#ifdef MLIR_TRT_ENABLE_CUDA
  llvm::dbgs() << "initializing cuda runtime\n";
  HANDLE_CUDART_ERROR(cudaFree(0), );
  HANDLE_CUDART_ERROR(cudaSetDevice(0), );
#endif
}
