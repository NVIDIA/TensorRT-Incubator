add_mlir_tensorrt_library(MLIRTensorRTCUDAToLLVM
  CUDAToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDADialect
    MLIRTensorRTLLVMConversionUtils
    MLIRTensorRTPlanToLLVM
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRRewrite
)
