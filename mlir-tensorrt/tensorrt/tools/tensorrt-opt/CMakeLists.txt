mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE TEST_LIBS OUT_VAR MTRTD_TEST_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR MTRTD_LIBS)
mtrt_add_tool(tensorrt-opt tensorrt-opt.cpp
  IGNORE_LINK_MTRT
  LINK_LIBS PRIVATE
  ${MTRTD_LIBS}
  ${MTRTD_TEST_LIBS}
  # TODO: this library is actually in the `compiler/` sub-project;
  # we need to refactor this library locaiton.
  MLIRTensorRTTestTensorKindAnalysis

  MLIR_LIBS PRIVATE
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIROptLib
  MLIRTensorDialect
  MLIRTransforms
  MLIRSCFDialect
)
