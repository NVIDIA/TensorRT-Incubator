add_mlir_tensorrt_library(
  MLIRTensorRTEmitCExtDataLayoutImpl
  DataLayoutImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIREmitCDialect
    MLIRDLTIDialect
  )
