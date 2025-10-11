set(sol2_includes "$<TARGET_PROPERTY:sol2::sol2,INTERFACE_INCLUDE_DIRECTORIES>")
include_directories(${sol2_includes})

add_subdirectory(Modules)

add_mlir_executor_runtime_library(MLIRTensorRTLuaExtensionRegistry
  LuaExtensionRegistry.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  lua::core
  MLIRTensorRTExecutorRuntimeAPI
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
)
