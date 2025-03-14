add_llvm_executable(executor-opt executor-opt.cpp)

llvm_update_compile_flags(executor-opt)

get_property(MLIR_EXECUTOR_TEST_LIBS GLOBAL PROPERTY MLIR_EXECUTOR_TEST_LIBS)

target_link_libraries(executor-opt PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  # TODO: Remove EmitC dialect dependency when we incorporate
  # https://github.com/llvm/llvm-project/pull/125466
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorTransforms
  MLIRTensorRTStdToExecutor
  MLIRTransforms
  ${MLIR_EXECUTOR_TEST_LIBS}
  )

mlir_check_all_link_libraries(executor-opt)
