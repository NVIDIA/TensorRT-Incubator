add_subdirectory(Modules)

add_mlir_executor_runtime_library(MLIRTensorRTLuaExtensionRegistry
  LuaExtensionRegistry.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI

  LINK_LIBS PRIVATE
  sol2::sol2
  lua::core
  )


add_mlir_executor_runtime_library(MLIRTensorRTExecutionEngineLuaRuntime
  LuaRuntime.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTLuaExtensionRegistry
  MLIRExecutorRuntimeCapabilities

  LINK_LIBS PRIVATE
  sol2::sol2
  lua::core
  nvtx3-cpp
  MLIRTensorRTExecutorRuntimeCommon
)

