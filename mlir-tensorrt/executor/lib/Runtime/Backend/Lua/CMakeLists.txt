add_subdirectory(Modules)

add_mlir_executor_runtime_library(MLIRTensorRTLuaExtensionRegistry
  LuaExtensionRegistry.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  lua::core
  MLIRTensorRTExecutorRuntimeAPI
  sol2::sol2
  )


get_property(runtime_capabilities GLOBAL PROPERTY MLIRExecutorRuntimeCapabilities)
add_mlir_executor_runtime_library(MLIRTensorRTExecutionEngineLuaRuntime
  LuaRuntime.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  lua::core
  ${runtime_capabilities}
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorRuntimeCommon
  MLIRTensorRTLuaExtensionRegistry
  nvtx3-cpp
  sol2::sol2
)
