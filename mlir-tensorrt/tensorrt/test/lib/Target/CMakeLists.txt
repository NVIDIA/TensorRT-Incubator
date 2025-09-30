set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
)

set(LLVM_LINK_COMPONENTS Support)

add_llvm_library(TensorRTTestPlugins
    TestPlugins.cpp TestV2Plugins.cpp

    MODULE

    BUILDTREE_ONLY

    TestPlugins.cpp TestV2Plugins.cpp
    LINK_LIBS PRIVATE
    ${MLIR_TRT_CUDA_TARGET}
    MLIRTensorRTCommonIncludes
    MLIRTRTTensorRTDynamicLoader
    TensorRTHeaderOnly
    )

include(GenerateExportHeader)
generate_export_header(TensorRTTestPlugins)
