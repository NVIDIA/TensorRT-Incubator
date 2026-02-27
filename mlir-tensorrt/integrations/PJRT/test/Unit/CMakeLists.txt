add_custom_target(MLIRTensorRTPJRTUnitTests)
set_target_properties(MLIRTensorRTPJRTUnitTests PROPERTIES FOLDER "MLIR-TensorRT Tests")

# Replace add_unittest with add_executable since we use find_package(GTest REQUIRED)
# GTest is always available, so we link to standard GTest targets
function(add_mlir_tensorrt_unittest name)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(${name}
    IGNORE_EXTERNALIZE_DEBUGINFO
    NO_INSTALL_RPATH
    PARTIAL_SOURCES_INTENDED
    ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "MLIR-TensorRT Tests/Unit")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  llvm_update_compile_flags(${name})
  # GTest is always available via find_package(GTest REQUIRED) in main CMakeLists.txt
  target_link_libraries(${name} PRIVATE
    GTest::gtest
    GTest::gtest_main
  )
  add_dependencies(MLIRTensorRTPJRTUnitTests ${name})
endfunction()


add_subdirectory(CAPI)

# Instantiate the LIT test suite configuration file. This ensures that
# tests under this directory will be correctly handled by the main config.
# Tests are still run by building the `check-mlir-tensorrt-pjrt` target.
# To only run tests under this directory, build the
# `check-mlir-tensorrt-pjrt-build-only` target and then use the
# `llvm-lit` command to run the tests with the `--filter=Unit` flag.
mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)
