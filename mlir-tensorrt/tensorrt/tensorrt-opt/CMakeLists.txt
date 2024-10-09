add_llvm_executable(tensorrt-opt tensorrt-opt.cpp)

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
  MLIRTensorRTTestTensorKindAnalysis
  MLIRSCFDialect
  )

mlir_check_all_link_libraries(tensorrt-opt)
