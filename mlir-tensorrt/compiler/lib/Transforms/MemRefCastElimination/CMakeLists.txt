add_mlir_tensorrt_library(MLIRTensorRTMemRefCastElimination
  MemRefCastElimination.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)