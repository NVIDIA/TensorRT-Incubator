add_mlir_tensorrt_library(MLIRTensorRTPlanToLLVM
  PlanToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanDialect
  MLIR_LIBS PUBLIC
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
  )
