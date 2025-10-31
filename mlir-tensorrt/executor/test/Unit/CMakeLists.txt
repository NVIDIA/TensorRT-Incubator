# This target can be used to build GTest-based unit tests.
add_custom_target(MLIRTensorRTExecutorUnitTests)

# Use this function for populating GTest-based unit tests.
function(add_mlir_executor_unittest target)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(${target} IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH PARTIAL_SOURCES_INTENDED ${ARGN})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  add_dependencies(MLIRTensorRTExecutorUnitTests ${target})
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

if ((TARGET gtest) OR (TARGET llvm_gtest))
    add_subdirectory(CAPI)
    add_subdirectory(Runtime)
    add_subdirectory(Utils)
else()
  message(WARNING "gtest not found, unittests will not be available")
endif()

mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)
