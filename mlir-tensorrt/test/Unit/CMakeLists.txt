# This target can be used to build GTest-based unit tests.
add_custom_target(MLIRTensorRTUnitTests)
set_target_properties(MLIRTensorRTUnitTests PROPERTIES FOLDER "MLIR-TensorRT Tests")

# Use this function for populating GTest-based unit tests.
function(add_mlir_tensorrt_unittest name)
  add_unittest(MLIRTensorRTUnitTests ${name} ${ARGN})
  _mtrt_set_target_compile_defs(${name})
  llvm_update_compiler_flags(compile_opts DISABLE_WERROR OFF)
endfunction()

include_directories(${MLIR_TENSORRT_ROOT_DIR}/compiler/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/compiler/include)
include_directories(${MLIR_TENSORRT_ROOT_DIR}/executor/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)

if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(CAPI)
    add_subdirectory(Runtime)
else()
  message(WARNING "gtest not found, unittests will not be available")
endif()
