mlir_executor_add_lua()
mlir_executor_add_sol2()

add_subdirectory(Modules)

set(_MLIR_TENSORRT_RUNTIME_MODULES
  MLIRTensorRTExecutorRuntimeCUDAModule
  MLIRTensorRTExecutorRuntimeCoreModule
  MLIRTensorRTExecutorRuntimeTensorRTModule
  MLIRTensorRTExecutorRuntimeNCCLModule
  MLIRTensorRTExecutorRuntimeCuBLASModule
)

add_mlir_library(MLIRTensorRTExecutionEngineLuaRuntime
  LuaRuntime.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  CUDA::cudart

  LINK_LIBS PRIVATE
  ${_MLIR_TENSORRT_RUNTIME_MODULES}
  sol2::sol2
  lua::core
  nvtx3-cpp
  MLIRTensorRTExecutorRuntimeCommon
)
