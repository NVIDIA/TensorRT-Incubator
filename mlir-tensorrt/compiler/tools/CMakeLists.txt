mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_TENSORRT_TEST_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE LIBS OUT_VAR MLIR_TENSORRT_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE DIALECT OUT_VAR MLIR_TENSORRT_DIALECT_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR MTRTD_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE TEST_LIBS OUT_VAR MTRTD_TEST_LIBS)


set(MLIR_TENSORRT_LIBS
  ${MLIR_TENSORRT_LIBS}
  ${MLIR_TENSORRT_DIALECT_LIBS}
  ${MTRTD_LIBS}
  ${MTRTD_TEST_LIBS}
  )
list(REMOVE_DUPLICATES MLIR_TENSORRT_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_subdirectory(mlir-tensorrt-opt)
add_subdirectory(mlir-tensorrt-compiler)
add_subdirectory(mlir-tensorrt-translate)
add_subdirectory(mlir-tensorrt-lsp-server)
add_subdirectory(mlir-tensorrt-runner)
