set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
)

unset(LLVM_LINK_COMPONENTS )

add_llvm_library(TensorRTTestPlugins
    TestPlugins.cpp TestV2Plugins.cpp

    SHARED

    TestPlugins.cpp TestV2Plugins.cpp
    LINK_LIBS PRIVATE
    MLIRTRTTensorRTDynamicLoader
    TensorRTHeaderOnly
    CUDA::cudart
    )

include(GenerateExportHeader)
generate_export_header(TensorRTTestPlugins)
