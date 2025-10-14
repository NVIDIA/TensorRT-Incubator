mtrt_get_project_targets(MLIRExecutor LIBRARY_TYPE LIBS OUT_VAR MLIR_EXECUTOR_LIBS)
mtrt_get_project_targets(MLIRExecutor LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_EXECUTOR_TEST_LIBS)

add_mlir_executor_tool(executor-opt executor-opt.cpp
  MLIR_LIBS PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRControlFlowTransforms

  LINK_LIBS PRIVATE
  ${MLIR_EXECUTOR_LIBS}
  ${MLIR_EXECUTOR_TEST_LIBS}
  )
