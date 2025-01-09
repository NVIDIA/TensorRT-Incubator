add_mlir_tensorrt_library(MLIRTensorRTCompilerTensorRTToExecutable
    TensorRTToExecutable.cpp
    TensorRTExtension.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTStablehloToExecutableIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    MLIRTensorRTCompilerClient
    )
