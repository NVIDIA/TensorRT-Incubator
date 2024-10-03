add_llvm_executable(tensorrt-opt tensorrt-opt.cpp)
get_property(MLIR_TENSORRT_DIALECT_TEST_LIBS GLOBAL PROPERTY MLIR_TENSORRT_DIALECT_TEST_LIBS)
llvm_update_compile_flags(tensorrt-opt)
target_link_libraries(tensorrt-opt PRIVATE
  MLIRTensorRTDialect
  MLIRTensorRTTransforms
  MLIRTensorRTTargetTensorRT
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIROptLib
  MLIRTensorDialect
  MLIRTransforms
  MLIRSCFDialect
  ${MLIR_TENSORRT_DIALECT_TEST_LIBS}
  MLIRTensorRTTestTensorKindAnalysis
  )

mlir_check_all_link_libraries(tensorrt-opt)
