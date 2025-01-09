add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StablehloToExecutable.cpp
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
    StablehloLinalgTransforms
    MLIRTensorRTCompilerClient
    )
