add_mlir_tensorrt_unittest(RuntimeCAPITests
  RuntimeCAPITests.cpp)

target_link_libraries(RuntimeCAPITests PUBLIC
  MLIRTensorRTCAPIRuntime
)
