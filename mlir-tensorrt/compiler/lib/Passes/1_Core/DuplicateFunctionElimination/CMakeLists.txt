add_mlir_tensorrt_library(MLIRTensorRTDuplicateFunctionElimination
    DuplicateFunctionElimination.cpp

    DEPENDS
    MLIRTensorRTPassesIncGenPassIncGen

    MLIR_LIBS PUBLIC
      MLIRPass
      MLIRFuncTransforms
)
