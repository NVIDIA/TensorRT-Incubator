# This target can be used to build GTest-based unit tests.
add_custom_target(MLIRKernelUnitTests)
set_target_properties(MLIRKernelUnitTests PROPERTIES FOLDER "MLIR-Kernel Unit Tests")

mtrt_get_project_targets(Kernel LIBRARY_TYPE LIBS OUT_VAR kernel_libs)


# Use this function for populating GTest-based unit tests.
function(add_mlir_kernel_unittest target)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(${target}
    PARTIAL_SOURCES_INTENDED
    IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH ${ARGN})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  add_dependencies(MLIRKernelUnitTests ${target})
  llvm_update_compile_flags(${target})
  if(TARGET gtest)
    target_link_libraries(${target} PRIVATE
     gtest gtest_main gmock)
  elseif(TARGET llvm_gtest)
    target_link_libraries(${target} PRIVATE
      llvm_gtest llvm_gtest_main)
  else()
    message(FATAL_ERROR "No GTest library found")
  endif()
endfunction()

if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
  add_mlir_kernel_unittest(MLIRKernelUtilsTests
    TilingUtilsTests.cpp
    GenerateSortTests.cpp
    PARTIAL_SOURCES_INTENDED
    )
  target_link_libraries(MLIRKernelUtilsTests PRIVATE
    MLIRTensorRTExecutorDialect
    ${kernel_libs}
  )
  mtrt_target_link_mlir_libraries(MLIRKernelUtilsTests PRIVATE
    MLIRAsmParser
    MLIRDialectUtils
    MLIRFuncInlinerExtension
    ${kernel_libs}
  )

else()
  message(FATAL_ERROR "gtest not found, unittests will not be available")
endif()
