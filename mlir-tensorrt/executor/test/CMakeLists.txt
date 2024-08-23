llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  LLVM_ENABLE_ASSERTIONS
  ${MLIR_EXECUTOR_FEATURE_FLAGS}
)

add_subdirectory(lib)

# All the GTest-based unit tests are located under 'Unit' subfolder.
add_subdirectory(Unit)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
get_property(MLIR_EXECUTOR_TEST_LIBS GLOBAL PROPERTY MLIR_EXECUTOR_TEST_LIBS)

set(MLIR_EXECUTOR_TEST_DEPENDS
  FileCheck count not
  executor-opt
  executor-translate
  executor-runner
  MLIRTensorRTExecutorUnitTests
  ${MLIR_EXECUTOR_TEST_LIBS}
  )

add_lit_testsuite(check-mlir-executor "Running the mlir-executor regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_EXECUTOR_TEST_DEPENDS}
  )
set_target_properties(check-mlir-executor PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_EXECUTOR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_EXECUTOR_TEST_DEPENDS})
add_custom_target(check-mlir-executor-build-only
  DEPENDS ${MLIR_EXECUTOR_TEST_DEPENDS}
)
