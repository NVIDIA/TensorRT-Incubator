mlir_executor_add_lua()
mlir_executor_add_sol2()

add_subdirectory(Modules)


add_mlir_executor_runtime_library(MLIRTensorRTExecutionEngineLuaRuntime
  LuaRuntime.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  CUDA::cudart

  LINK_LIBS PRIVATE
  MLIRExecutorRuntimeCapabilities
  sol2::sol2
  lua::core
  nvtx3-cpp
  MLIRTensorRTExecutorRuntimeCommon
)
