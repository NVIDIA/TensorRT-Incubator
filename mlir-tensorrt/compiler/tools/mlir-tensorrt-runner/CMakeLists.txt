add_llvm_executable(mlir-tensorrt-runner
  PARTIAL_SOURCES_INTENDED
  mlir-tensorrt-runner.cpp
  )
target_link_libraries(mlir-tensorrt-runner PRIVATE
  MLIRExecutorRunnerLib
  )
mlir_check_link_libraries(mlir-tensorrt-runner)
