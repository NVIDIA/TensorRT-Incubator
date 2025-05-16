add_llvm_executable(mlir-tensorrt-runner
PARTIAL_SOURCES_INTENDED
mlir-tensorrt-runner.cpp)
target_link_libraries(mlir-tensorrt-runner PRIVATE
MLIRExecutorRunnerLib
)
target_include_directories(mlir-tensorrt-runner PRIVATE
${MLIR_TENSORRT_ROOT_DIR}/include
${MLIR_TENSORRT_ROOT_DIR}/executor/include)
mlir_check_link_libraries(mlir-tensorrt-runner)
