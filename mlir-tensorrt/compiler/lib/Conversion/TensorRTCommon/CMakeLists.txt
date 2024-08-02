add_mlir_tensorrt_library(MLIRTensorRTConvertToTensorRTCommon
  ConvertToTensorRTCommon.cpp  

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTensorRTDialect
  MLIRTransforms
)