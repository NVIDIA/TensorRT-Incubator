add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
    Client.cpp
    Extension.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
)

add_subdirectory(TensorRTExtension)

add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StableHloToExecutable.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
    MLIRTensorRTCompilerClient
    MLIRTensorRTCompilerTensorRTExtension
)
