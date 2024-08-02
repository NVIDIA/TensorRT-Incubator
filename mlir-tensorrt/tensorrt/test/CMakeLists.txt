add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_TENSORRT_TEST_DEPENDS
  FileCheck count not
  tensorrt-opt
  TensorRTTestPlugins
  )

add_lit_testsuite(check-mlir-tensorrt-dialect "Running the mlir-tensorrt-dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS}
  )
set_target_properties(check-mlir-tensorrt-dialect PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_TENSORRT_DIALECT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS})
