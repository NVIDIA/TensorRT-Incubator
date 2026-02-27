# This target can be used to build GTest-based unit tests.
add_custom_target(MLIRTensorRTExecutorUnitTests)

# Use this function for populating GTest-based unit tests.
function(add_mlir_executor_unittest target)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(${target} IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH PARTIAL_SOURCES_INTENDED ${ARGN})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  add_dependencies(MLIRTensorRTExecutorUnitTests ${target})
  llvm_update_compile_flags(${target})
  # GTest is always available via find_package(GTest REQUIRED) in main CMakeLists.txt
  target_link_libraries(${target} PRIVATE
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
  )
endfunction()

# GTest is always available, so always add subdirectories
add_subdirectory(CAPI)
add_subdirectory(Runtime)
add_subdirectory(Utils)

mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)
