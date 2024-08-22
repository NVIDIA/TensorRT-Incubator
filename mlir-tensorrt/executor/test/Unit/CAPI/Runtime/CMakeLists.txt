add_mlir_executor_unittest(RuntimeCAPITests RuntimeCAPITests.cpp)
target_link_libraries(RuntimeCAPITests PUBLIC
  MLIRTensorRTCAPIRuntime
  )
