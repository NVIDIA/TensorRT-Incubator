add_mlir_tensorrt_library(MLIRTensorRTCompilerTensorRTExtension
    TensorRTExtension.cpp

    LINK_LIBS PUBLIC
    MLIRTensorRTCompilerClient
    MLIRTensorRTTargetTensorRT
    MLIRFuncDialect
    )