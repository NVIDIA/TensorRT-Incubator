add_mlir_tensorrt_unittest(Int4Tests
  Int4Tests.cpp)
target_include_directories(Int4Tests PUBLIC
  $<BUILD_INTERFACE:${MLIR_TENSORRT_ROOT_DIR}/executor/include>)
