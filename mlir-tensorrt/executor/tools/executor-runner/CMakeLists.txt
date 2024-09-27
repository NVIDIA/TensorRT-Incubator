set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(executor-runner
  executor-runner.cpp
  )
llvm_update_compile_flags(executor-runner)
target_link_libraries(executor-runner
  PRIVATE
  MLIRExecutorRunnerLib
  )

mlir_check_link_libraries(executor-runner)
