set(MLIR_TENSORRT_COMPILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(MLIR_TENSORRT_COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PARENT_SCOPE)

add_library(MLIRTensorRTCompilerIncludes INTERFACE)
target_include_directories(MLIRTensorRTCompilerIncludes INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  )

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

get_target_property(common_include_dirs
                    MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(executor_include_dirs
                    MLIRExecutorIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${common_include_dirs} ${executor_include_dirs})

# Because MLIR has a level of indirection that lets implementation for
# interfaces be provided by separate implementation code
# ("PromisedInterfaces/ExternalModels") which is registered at runtime, it is
# difficult to capture all dependencies for dialects we require in the compiler
# purely through target dependency properties. To see what we require from
# usptream, look at the file `mlir-tensorrt/Compiler/InitAllDialects.h`. Therefore, we
# manually enumerate some dependencies here, mainly for providing the functions
# registering interface external models.
mtrt_add_project_targets(MLIRTensorRT
  LIBRARY_TYPE LIBS
  TARGETS
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRAsyncDialect
  MLIRBufferizationTransforms
  MLIRControlFlowTransforms
  MLIREmitCExtDataLayoutImpl
  MLIRNVVMTarget
  MLIRPtrDialect
  MLIRTargetLLVM
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransformOps
  MLIRTargetCpp
  MLIRDebug
  )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
