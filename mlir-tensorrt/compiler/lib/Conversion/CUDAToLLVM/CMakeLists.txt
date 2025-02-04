add_mlir_tensorrt_library(MLIRTensorRTCUDAToLLVM
  CUDAToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRRewrite
  MLIRTensorRTCUDADialect
  MLIRTensorRTLLVMConversionUtils
  MLIRTensorRTPlanToLLVM
)
