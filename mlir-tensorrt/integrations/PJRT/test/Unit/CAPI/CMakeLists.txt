set(LLVM_LINK_COMPONENTS )

add_mlir_tensorrt_unittest(PJRTPluginLoadTests PJRTPluginLoadTest.cpp)
target_link_libraries(PJRTPluginLoadTests PRIVATE dl)
target_compile_definitions(PJRTPluginLoadTests PRIVATE
  "MLIR_TENSORRT_PJRT_PLUGIN_SO_PATH=\"$<TARGET_FILE:mlir-tensorrt-pjrt>\""
)

add_mlir_tensorrt_unittest(PJRTCAPITests PJRTCAPITests.cpp)
target_link_libraries(PJRTCAPITests PRIVATE
  mlir-tensorrt-pjrt
  MLIRTensorRTSupportStatus)
