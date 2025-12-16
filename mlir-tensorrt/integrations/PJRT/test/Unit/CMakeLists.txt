if (NOT EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
  message(FATAL_ERROR "gtest not found, unittests will not be available")
endif()

add_custom_target(MLIRTensorRTPJRTUnitTests)
set_target_properties(MLIRTensorRTPJRTUnitTests PROPERTIES FOLDER "MLIR-TensorRT Tests")

function(add_mlir_tensorrt_unittest name)
  add_unittest(MLIRTensorRTPJRTUnitTests ${name}
    PARTIAL_SOURCES_INTENDED
    ${ARGN})
  set_target_properties(MLIRTensorRTPJRTUnitTests PROPERTIES FOLDER "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()


add_subdirectory(CAPI)

# Instantiate the LIT test suite configuration file. This ensures that
# tests under this directory will be correctly handled by the main config.
# Tests are still run by building the `check-mlir-tensorrt-pjrt` target.
# To only run tests under this directory, build the
# `check-mlir-tensorrt-pjrt-build-only` target and then use the
# `llvm-lit` command to run the tests with the `--filter=Unit` flag.
mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)
