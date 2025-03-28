add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
    Client.cpp
    Extension.cpp
    OptionsProviders.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTTargetLua
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
    MLIRTensorRTSupportDeviceInfo
)

add_subdirectory(StablehloToExecutable)
add_subdirectory(TensorRTToExecutable)
