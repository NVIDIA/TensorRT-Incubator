add_mlir_tensorrt_library(MLIRTensorRTDuplicateFunctionElimination
    DuplicateFunctionElimination.cpp

    DEPENDS
    MLIRTensorRTGenericTransformPassIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRFuncTransforms
)

