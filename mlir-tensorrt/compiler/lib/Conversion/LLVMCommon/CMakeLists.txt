add_mlir_tensorrt_library(MLIRTensorRTLLVMConversionUtils
  LLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)