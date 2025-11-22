add_mlir_tensorrt_library(MLIRTensorRTLLVMConversionUtils
  LLVMCommon.cpp

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)