add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
    Client.cpp
    Extension.cpp
    OptionsRegistry.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
)

add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StableHloToExecutable.cpp
    # TODO: TensorRTExtension should be an independent library.
    TensorRTExtension/TensorRTExtension.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
    MLIRTensorRTCompilerClient
    )
