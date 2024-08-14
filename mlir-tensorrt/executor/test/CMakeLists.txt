llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  LLVM_ENABLE_ASSERTIONS
  ${MLIR_EXECUTOR_FEATURE_FLAGS}
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_EXECUTOR_TEST_DEPENDS
  FileCheck count not
  executor-opt
  executor-translate
  executor-runner
  )

add_lit_testsuite(check-mlir-executor "Running the mlir-executor regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_EXECUTOR_TEST_DEPENDS}
  )
set_target_properties(check-mlir-executor PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_EXECUTOR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_EXECUTOR_TEST_DEPENDS})
