llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  ENABLE_ASAN
  ${MLIR_TRT_FEATURE_FLAGS}
)

add_subdirectory(Unit)
add_subdirectory(lib)

mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)

# Add the LIT test suite configuration for discovering all the GTest based
# executables located under the `Unit` folder.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )


set(MLIR_KERNEL_TEST_DEPENDS
  FileCheck count not
  kernel-opt
  kernel-translate
  MLIRKernelUnitTests
  )
# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   list(APPEND MLIR_KERNEL_TEST_DEPENDS MLIRTensorRTPythonKernelCompilerModules)
# endif()

add_lit_testsuite(check-mlir-kernel "Running the mlir-kernel regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_KERNEL_TEST_DEPENDS}
  )
set_target_properties(check-mlir-kernel PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_KERNEL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_KERNEL_TEST_DEPENDS})
add_custom_target(check-mlir-kernel-build-only
  DEPENDS ${MLIR_KERNEL_TEST_DEPENDS})

if(TARGET check-all-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt check-mlir-kernel)
  add_dependencies(check-all-mlir-tensorrt-build-only check-mlir-kernel-build-only)
endif()
