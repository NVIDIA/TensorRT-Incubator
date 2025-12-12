add_mlir_tensorrt_library(MLIRTensorRTTensorExtPadToInsertSlice
  TensorExtPadToInsertSlice.cpp

  DEPENDS
  MLIRTensorRTGenericTransformPassIncGen

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRTensorDialect
  )
