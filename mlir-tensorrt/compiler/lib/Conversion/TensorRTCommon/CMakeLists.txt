add_mlir_tensorrt_library(MLIRTensorRTConvertToTensorRTCommon
  ConvertToTensorRTCommon.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTransforms
  )
