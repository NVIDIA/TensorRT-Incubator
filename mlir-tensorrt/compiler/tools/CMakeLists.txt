# Below we accumulate a big list of project libraries in
#`MLIR_TENSORRT_LIBS` that tools like `mlir-tensorrt-opt` and `mlir-tensorrt-compiler`
# will need to link against.
# We need to be careful to link against libMTRT when
# MLIR_TRT_LINK_MTRT_DYLIB is ON, otherwise we link against all the
# individual static libraries.

# Test libraries are always excluded from libMTRT, so we always link
# those directly.
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_TENSORRT_TEST_LIBS)
set(MLIR_TENSORRT_LIBS
  ${MLIR_TENSORRT_TEST_LIBS}
)

if(MLIR_TRT_LINK_MTRT_DYLIB)
  list(APPEND MLIR_TENSORRT_LIBS MTRT)
else()
  mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE LIBS OUT_VAR MTRT_MAIN_LIBS)
  mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE MLIR_LIBS OUT_VAR MTRT_MLIR_LIBS)
  mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE DIALECT OUT_VAR MTRT_DIALECT_LIBS)
  mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR TRT_DIALECT_LIBS)
  list(APPEND MLIR_TENSORRT_LIBS
    ${MTRT_MAIN_LIBS}
    ${MTRT_MLIR_LIBS}
    ${MTRT_DIALECT_LIBS}
    ${TRT_DIALECT_LIBS}
    )
endif()

list(REMOVE_DUPLICATES MLIR_TENSORRT_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
  )
if(MLIR_TRT_LINK_MTRT_DYLIB)
  set(LLVM_LINK_COMPONENTS )
endif()

add_subdirectory(mlir-tensorrt-shlib)
add_subdirectory(mlir-tensorrt-opt)
add_subdirectory(mlir-tensorrt-compiler)
add_subdirectory(mlir-tensorrt-translate)
add_subdirectory(mlir-tensorrt-lsp-server)
add_subdirectory(mlir-tensorrt-runner)
