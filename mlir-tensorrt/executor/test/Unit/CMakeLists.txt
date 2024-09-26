# This target can be used to build GTest-based unit tests.
add_custom_target(MLIRTensorRTExecutorUnitTests)
set_target_properties(MLIRTensorRTExecutorUnitTests PROPERTIES FOLDER "MLIR-TensorRT Executor Unit Tests")

# Use this function for populating GTest-based unit tests.
function(add_mlir_executor_unittest name)
  add_unittest(MLIRTensorRTExecutorUnitTests ${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(CAPI)
    add_subdirectory(Runtime)
    add_subdirectory(Utils)
else()
  message(WARNING "gtest not found, unittests will not be available")
endif()

# Add the LIT test suite configuration for discovering all the GTest based
# executables located under the `Unit` folder.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

