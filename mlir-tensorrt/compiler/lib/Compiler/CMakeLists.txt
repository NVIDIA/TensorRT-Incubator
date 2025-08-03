add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
    Client.cpp
    Extension.cpp
    OptionsProviders.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRDebug
    MLIRIR
    MLIRPass
    MLIRTargetCpp
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTSupportDeviceInfo
    MLIRTensorRTTargetLua

)

add_subdirectory(StablehloToExecutable)
add_subdirectory(TensorRTToExecutable)
