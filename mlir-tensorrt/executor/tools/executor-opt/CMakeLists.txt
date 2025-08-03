add_llvm_executable(executor-opt executor-opt.cpp)

llvm_update_compile_flags(executor-opt)
mtrt_get_project_targets(MLIRExecutor LIBRARY_TYPE LIBS OUT_VAR MLIR_EXECUTOR_LIBS)
mtrt_get_project_targets(MLIRExecutor LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_EXECUTOR_TEST_LIBS)

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
  MLIRTransforms
  ${MLIR_EXECUTOR_LIBS}
  ${MLIR_EXECUTOR_TEST_LIBS}
  )

mlir_check_all_link_libraries(executor-opt)
