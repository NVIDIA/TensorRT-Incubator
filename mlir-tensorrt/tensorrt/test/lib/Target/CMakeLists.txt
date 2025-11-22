set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(TensorRTTestPlugins MODULE
    TestPlugins.cpp TestV2Plugins.cpp)
set_target_properties(TensorRTTestPlugins PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME TensorRTTestPlugins
    LIBRARY_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
    RUNTIME_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
    PREFIX ""
)
llvm_update_compile_flags(TensorRTTestPlugins)

target_link_libraries(TensorRTTestPlugins PRIVATE
    ${MLIR_TRT_CUDA_TARGET}
    MLIRTensorRTCommonIncludes
    TensorRTHeaderOnly
    )
include(GenerateExportHeader)
generate_export_header(TensorRTTestPlugins)

if(MLIR_TRT_LINK_MTRT_DYLIB)
  target_link_libraries(TensorRTTestPlugins PRIVATE MTRT)
endif()

if(NOT MLIR_TRT_LINK_MTRT_DYLIB OR BUILD_SHARED_LIBS)
  target_link_libraries(TensorRTTestPlugins PRIVATE
    TensorRTDynamicLoader
    LLVMSupport)
endif()
