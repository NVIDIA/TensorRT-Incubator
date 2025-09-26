add_mlir_tensorrt_library(
  MLIREmitCExtDataLayoutImpl
  DataLayoutImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIREmitCDialect
    MLIRDLTIDialect
  )
