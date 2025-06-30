set(MLIR_TENSORRT_COMPILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")


set(MLIR_TENSORRT_COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PARENT_SCOPE)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

# We use the MLIR_TENSORRT_LIBS global property to aggregate the list of all
# declared compiler libraries. This is helpful for simplifying the link
# dependency declarations for tools that must link "the world" like
# `mlir-tensorrt-opt`.

# Because MLIR has a level of indirection that lets implementation for
# interfaces be provided by separate implementation code
# ("PromisedInterfaces/ExternalModels") which is registered at runtime, it is
# difficult to capture all dependencies for dialects we require in the compiler
# purely through target dependency properties. To see what we require from
# usptream, look at the file `mlir-tensorrt/InitAllDialects.h`. Therefore, we
# manually enumerate some dependencies here, mainly for providing the functions
# registering interface external models.
add_mlir_tensorrt_compiler_dependency(MLIRArithTransforms)
add_mlir_tensorrt_compiler_dependency(MLIRArithValueBoundsOpInterfaceImpl)
add_mlir_tensorrt_compiler_dependency(MLIRAsyncDialect)
add_mlir_tensorrt_compiler_dependency(MLIRBufferizationTransforms)
add_mlir_tensorrt_compiler_dependency(MLIRControlFlowTransforms)
add_mlir_tensorrt_compiler_dependency(MLIRNVVMTarget)
add_mlir_tensorrt_compiler_dependency(MLIRPtrDialect)
add_mlir_tensorrt_compiler_dependency(MLIRTargetLLVM)
add_mlir_tensorrt_compiler_dependency(MLIRTensorTransformOps)
add_mlir_tensorrt_compiler_dependency(MLIREmitCExtDataLayoutImpl)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

