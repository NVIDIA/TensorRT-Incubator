add_mlir_tensorrt_library(MLIRTensorRTLowerLinalgCopies
  LowerLinalgCopies.cpp

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
)