add_library(MLIRExecutorRuntimeCapabilities INTERFACE)
macro(add_mlir_executor_runtime_capability_library name)
    add_mlir_executor_runtime_library(${name} ${ARGN})
    target_link_libraries(MLIRExecutorRuntimeCapabilities INTERFACE ${name})
endmacro()

add_subdirectory(Core)
add_subdirectory(CUDA)
add_subdirectory(TensorRT)
add_subdirectory(Utils)

if(MLIR_EXECUTOR_ENABLE_NCCL)
    add_subdirectory(NCCL)
endif()

if(MLIR_EXECUTOR_ENABLE_CUBLAS)
    add_subdirectory(CuBLAS)
endif()

add_mlir_library_install(MLIRExecutorRuntimeCapabilities)