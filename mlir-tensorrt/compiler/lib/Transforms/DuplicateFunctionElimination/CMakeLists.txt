add_mlir_tensorrt_library(MLIRTensorRTDuplicateFunctionElimination
    DuplicateFunctionElimination.cpp

    DEPENDS
    MLIRTensorRTGenericTransformPassIncGen

    MLIR_LIBS PUBLIC
      MLIRPass
      MLIRFuncTransforms
)

