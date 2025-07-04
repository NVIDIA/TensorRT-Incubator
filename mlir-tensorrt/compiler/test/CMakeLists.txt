enable_testing()

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  ${MLIR_TRT_FEATURE_FLAGS}
  )

# Add test libraries.
add_subdirectory(lib)

get_property(MLIR_TENSORRT_TEST_LIBS GLOBAL PROPERTY MLIR_TENSORRT_TEST_LIBS)

# Instantiate the main LIT test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_TENSORRT_TEST_DEPENDS_
  FileCheck count not
  mlir-tensorrt-opt
  mlir-tensorrt-runner
  mlir-tensorrt-translate
  ${MLIR_TENSORRT_TEST_LIBS}
)

if(MLIR_TRT_TARGET_TENSORRT AND
  (MLIR_TRT_TENSORRT_VERSION VERSION_GREATER_EQUAL 10.0))
  list(APPEND MLIR_TENSORRT_TEST_DEPENDS_
    TensorRTTestPlugins)
endif()

if(MLIR_TRT_ENABLE_PYTHON)
  list(APPEND MLIR_TENSORRT_TEST_DEPENDS_

    # Ensure all Python packages are built before running python tests.
    # The compiler package is added conditional on
    MLIRTensorRTPythonToolsModules
    MLIRTensorRTPythonCompilerModules
    MLIRTensorRTPythonRuntimeModules)
endif()

add_lit_testsuite(check-mlir-tensorrt "Running the mlir-tensorrt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS_}
)
set_target_properties(check-mlir-tensorrt PROPERTIES FOLDER "Tests")
add_lit_testsuites(MLIR_TENSORRT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS_})
add_custom_target(check-mlir-tensorrt-build-only
  DEPENDS ${MLIR_TENSORRT_TEST_DEPENDS_}
)

if(TARGET check-all-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt check-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt-build-only check-mlir-tensorrt-build-only)
endif()
