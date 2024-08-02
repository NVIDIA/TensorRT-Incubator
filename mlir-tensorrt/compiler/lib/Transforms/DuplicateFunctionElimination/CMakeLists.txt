add_mlir_tensorrt_library(MLIRTensorRTDuplicateFunctionElimination
    DuplicateFunctionElimination.cpp

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRFuncTransforms
)

