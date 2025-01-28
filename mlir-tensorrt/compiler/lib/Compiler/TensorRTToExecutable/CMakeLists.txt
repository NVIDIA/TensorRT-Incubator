add_mlir_tensorrt_library(MLIRTensorRTCompilerTensorRTToExecutable
    TensorRTToExecutable.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTTensorRTToExecutableIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    MLIRTensorRTCompilerClient
    )
