llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  LLVM_ENABLE_ASSERTIONS
  ${MLIR_TRT_FEATURE_FLAGS}
)

add_subdirectory(lib)

# All the GTest-based unit tests are located under 'Unit' subfolder.
add_subdirectory(Unit)

mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)

mtrt_get_project_targets(MLIRExecutor LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_EXECUTOR_TEST_LIBS)

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

if(TARGET check-all-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt check-mlir-executor)
  add_dependencies(check-all-mlir-tensorrt-build-only check-mlir-executor-build-only)
endif()
