set(MLIR_TENSORRT_COMPILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLIR_COMPILER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_COMPILER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(MLIR_TENSORRT_COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PARENT_SCOPE)

add_library(MLIRTensorRTCompilerIncludes INTERFACE)
target_include_directories(MLIRTensorRTCompilerIncludes INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  )
mtrt_add_install(MLIRTensorRTCompilerIncludes UMBRELLA mtrt-headers)

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
  LIBRARY_TYPE MLIR_LIBS
  TARGETS
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRAsyncDialect
  MLIRBufferizationTransforms
  MLIRControlFlowTransforms
  MLIRDebug
  MLIRFuncInlinerExtension
  MLIRNVVMTarget
  MLIRPtrDialect
  MLIRTargetCpp
  MLIRTargetLLVM
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransformOps
  MLIRTransformDialectTransforms
  )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

# Tools must be sequenced after tests.
add_subdirectory(tools)

# Add header installation components
set(MTRT_COMPILER_HEADER_DIRS
  "include/mlir-tensorrt"
  "include/mlir-tensorrt-c"
  "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt"
  "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt-c"
  # For the Features.h header.
  "${PROJECT_BINARY_DIR}/include/mlir-tensorrt"
)

mtrt_add_header_installation_components(mtrt-headers
  DIRECTORIES
   ${MTRT_COMPILER_HEADER_DIRS}
  )
