set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_executor_tool(executor-runner executor-runner.cpp
  LINK_LIBS PRIVATE
  MLIRExecutorRunnerLib
  )
