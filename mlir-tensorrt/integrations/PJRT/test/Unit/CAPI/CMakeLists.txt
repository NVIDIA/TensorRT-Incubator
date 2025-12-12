set(LLVM_LINK_COMPONENTS )
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

add_mlir_tensorrt_unittest(PJRTPluginLoadTests PJRTPluginLoadTest.cpp)
target_link_libraries(PJRTPluginLoadTests PRIVATE dl)
target_compile_definitions(PJRTPluginLoadTests PRIVATE
  "MLIR_TENSORRT_PJRT_PLUGIN_SO_PATH=\"$<TARGET_FILE:mlir-tensorrt-pjrt>\""
)

add_mlir_tensorrt_unittest(PJRTCAPITests PJRTCAPITests.cpp)
# All gtest-based LLVM tests link LLVMSupport, which also has symbols
# in the `mlir-tensorrt-pjrt.so` shlib. So to avoid ODR violations,
# we just link the static version for the unit tests.
target_link_libraries(PJRTCAPITests PRIVATE
  mlir-tensorrt-pjrt-static
  MLIRTensorRTSupportStatus
  )
