add_mlir_tensorrt_dialect_test_library(MLIRTensorRTTestTypeInferencePass
    TestTypeInferencePass.cpp

    LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransformUtils
    )
