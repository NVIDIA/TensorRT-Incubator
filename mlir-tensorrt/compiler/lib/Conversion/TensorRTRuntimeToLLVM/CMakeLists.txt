add_mlir_tensorrt_library(MLIRTensorRTTensorRTRuntimeToLLVM
  TensorRTRuntimeToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRRewrite
  MLIRTensorRTCUDAToLLVM
  MLIRTensorRTLLVMConversionUtils
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTPlanToLLVM
  )
