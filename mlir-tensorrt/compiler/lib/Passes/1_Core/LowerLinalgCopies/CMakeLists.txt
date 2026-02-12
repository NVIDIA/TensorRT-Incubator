add_mlir_tensorrt_library(MLIRTensorRTLowerLinalgCopies
  LowerLinalgCopies.cpp

  MLIR_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRTransformUtils
)