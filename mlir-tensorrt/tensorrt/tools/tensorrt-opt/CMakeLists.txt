add_llvm_executable(tensorrt-opt tensorrt-opt.cpp)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE TEST_LIBS OUT_VAR MTRTD_TEST_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR MTRTD_LIBS)
llvm_update_compile_flags(tensorrt-opt)
target_link_libraries(tensorrt-opt PRIVATE
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIROptLib
  MLIRTensorDialect
  MLIRTransforms
  MLIRSCFDialect
  ${MTRTD_LIBS}
  ${MTRTD_TEST_LIBS}
  # TODO: this library is actually in the `compiler/` sub-project;
  # we need to refactor this library locaiton.
  MLIRTensorRTTestTensorKindAnalysis
  )

mlir_check_all_link_libraries(tensorrt-opt)
