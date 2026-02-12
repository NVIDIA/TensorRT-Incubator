add_mlir_tensorrt_library(MLIRTensorRTPlanToLLVM
  PlanToLLVM.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTPlanDialect
  MLIR_LIBS PUBLIC
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
  )
