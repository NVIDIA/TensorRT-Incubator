add_library(MLIRExecutorRuntimeCapabilities INTERFACE)
macro(add_mlir_executor_runtime_capability_library name)
    add_mlir_executor_runtime_library(${name} ${ARGN})
    target_link_libraries(MLIRExecutorRuntimeCapabilities INTERFACE ${name})
endmacro()

add_subdirectory(Utils)

add_subdirectory(Core)

if(MLIR_TRT_ENABLE_CUDA)
    add_subdirectory(CUDA)
endif()

if(MLIR_TRT_TARGET_TENSORRT)
    add_subdirectory(TensorRT)
endif()

if(MLIR_TRT_ENABLE_NCCL)
    add_subdirectory(NCCL)
endif()

if(MLIR_TRT_ENABLE_CUBLAS)
    add_subdirectory(CuBLAS)
endif()

mtrt_add_install(MLIRExecutorRuntimeCapabilities)
