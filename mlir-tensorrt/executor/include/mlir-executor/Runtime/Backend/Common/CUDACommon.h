#ifndef MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_CUDACOMMON
#define MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_CUDACOMMON
#ifdef MLIR_TRT_ENABLE_CUDA

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mlir-executor/Runtime/API/API.h"

namespace mlirtrt::runtime {

/// Stream a CUDA driver error to an ostream.
std::ostream &operator<<(std::ostream &es, CUresult result);
/// Stream a CUDA runtime error to an ostream.
std::ostream &operator<<(std::ostream &es, cudaError_t error);

template <typename T>
struct PointerWrapper {

  PointerWrapper(uintptr_t ptr) : ptr(ptr) {}
  PointerWrapper(T ptr) : ptr(reinterpret_cast<uintptr_t>(ptr)) {}

  operator T() { return reinterpret_cast<T>(ptr); }
  operator uintptr_t() { return ptr; }

  uintptr_t ptr;
};

struct CudaEventPtr : public PointerWrapper<cudaEvent_t> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaEventPtr> create(ResourceTracker &tracker);
};

struct CudaModulePtr : public PointerWrapper<CUmodule> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaModulePtr>
  create(ResourceTracker &tracker, llvm::StringRef ptx, llvm::StringRef arch);
};

struct CudaFunctionPtr : public PointerWrapper<CUfunction> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaFunctionPtr>
  create(ResourceTracker &tracker, CudaModulePtr module, llvm::StringRef name);
};

} // namespace mlirtrt::runtime

#endif // MLIR_TRT_ENABLE_CUDA
#endif // MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_CUDACOMMON
