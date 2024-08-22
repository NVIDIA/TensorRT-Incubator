add_llvm_executable(executor-opt executor-opt.cpp)

llvm_update_compile_flags(executor-opt)

get_property(MLIR_EXECUTOR_TEST_LIBS GLOBAL PROPERTY MLIR_EXECUTOR_TEST_LIBS)

target_link_libraries(executor-opt PRIVATE
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorTransforms
  MLIRTensorRTStdToExecutor
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRTransforms
  MLIROptLib
  MLIRSCFToControlFlow
  MLIRSCFDialect
  MLIRControlFlowDialect
  ${MLIR_EXECUTOR_TEST_LIBS}
  )

mlir_check_all_link_libraries(executor-opt)
