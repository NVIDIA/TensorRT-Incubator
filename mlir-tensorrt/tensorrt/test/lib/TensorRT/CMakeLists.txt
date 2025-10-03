add_mlir_tensorrt_dialect_test_library(MLIRTensorRTTestTypeInferencePass
    TestTypeInferencePass.cpp

    MLIR_LIBS PUBLIC
    MLIRTensorDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransformUtils

    LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
    )
