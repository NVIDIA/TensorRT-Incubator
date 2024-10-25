get_property(MLIR_TENSORRT_TEST_LIBS GLOBAL PROPERTY MLIR_TENSORRT_TEST_LIBS)
get_property(MLIR_TENSORRT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_LIBS)
get_property(MLIR_TENSORRT_DIALECT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_DIALECT_LIBS)
set(MLIR_TENSORRT_LIBS ${MLIR_TENSORRT_LIBS} ${MLIR_TENSORRT_DIALECT_LIBS} ${MLIR_TENSORRT_TEST_LIBS})
list(REMOVE_DUPLICATES MLIR_TENSORRT_LIBS)
include_directories("${MLIR_TENSORRT_ROOT_DIR}/executor/include")
include_directories("${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include")

# Only build the benchmark if Google benchmark is available.
if(TARGET benchmark)
  add_executable(mlir-tensorrt-clustering-benchmark
    ClusteringBenchmarkMain.cpp
    )
  target_link_libraries(mlir-tensorrt-clustering-benchmark PRIVATE
    ${MLIR_TENSORRT_LIBS}
    MLIRExecutorGenericClustering
    benchmark
    )
  set_target_properties(mlir-tensorrt-clustering-benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MLIR_TENSORRT_ROOT_BINARY_DIR}/bin"
    )
  _mtrt_set_target_compile_defs(mlir-tensorrt-clustering-benchmark)
  llvm_update_compile_flags(mlir-tensorrt-clustering-benchmark)
endif()
