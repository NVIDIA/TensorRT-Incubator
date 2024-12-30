add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
    Client.cpp
    Extension.cpp
    OptionsRegistry.cpp
    OptionsProviders.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorRTRegistration
    MLIRTensorRTTargetLua
    MLIRTensorRTOptionUtils
    MLIRTensorRTTargetTensorRT
    StablehloLinalgTransforms
    MLIRTensorRTSupportDeviceInfo
)

add_subdirectory(StablehloToExecutable)