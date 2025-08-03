add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  ${MLIR_TRT_DIALECT_FEATURE_FLAGS}
)

mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)

set(MLIR_TENSORRT_TEST_DEPENDS
  FileCheck count not
  tensorrt-opt
  )

if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND MLIR_TENSORRT_TEST_DEPENDS TensorRTTestPlugins)
endif()

add_lit_testsuite(check-mlir-tensorrt-dialect "Running the mlir-tensorrt-dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS}
  )
set_target_properties(check-mlir-tensorrt-dialect PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_TENSORRT_DIALECT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS})

add_custom_target(check-mlir-tensorrt-dialect-build-only
  DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS}
)

if(TARGET check-all-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt check-mlir-tensorrt-dialect)
  add_dependencies(check-all-mlir-tensorrt-build-only check-mlir-tensorrt-dialect-build-only)
endif()
