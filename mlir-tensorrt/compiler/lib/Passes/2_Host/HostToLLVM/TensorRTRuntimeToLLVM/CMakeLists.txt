add_mlir_tensorrt_library(MLIRTensorRTTensorRTRuntimeToLLVM
  TensorRTRuntimeToLLVM.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDAToLLVM
    MLIRTensorRTLLVMConversionUtils
    MLIRTensorRTTensorRTRuntimeDialect
    MLIRTensorRTPlanToLLVM
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRRewrite
  )
