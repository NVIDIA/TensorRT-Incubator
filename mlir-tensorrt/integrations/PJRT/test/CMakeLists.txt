
# ==== Canonicalize boolean build flags =====================================
llvm_canonicalize_cmake_booleans(
  LLVM_ENABLE_ASSERTIONS
  ENABLE_ASAN
  ${MLIR_TRT_FEATURE_FLAGS}
  )

# ==== Add unit test subfolder ==============================================

add_subdirectory(Unit)


# ==== Configure lit.site.cfg.py.in template ================================

# Instantiate the main LIT test suite.
mtrt_configure_lit_test_site_config(lit.site.cfg.py.in)

# ==== Setup test dependencies =============================================

set(test_depends_
  FileCheck count not
  mlir-tensorrt-pjrt
  MLIRTensorRTPJRTUnitTests
  MLIRTensorRTPythonToolsModules
  )

add_lit_testsuite(check-mlir-tensorrt-pjrt "Running the mlir-tensorrt-pjrt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${test_depends_}
  )
set_target_properties(check-mlir-tensorrt-pjrt PROPERTIES FOLDER "Tests")
add_custom_target(check-mlir-tensorrt-pjrt-build-only
  DEPENDS ${test_depends_}
)

# ==== Add pytest targets for JAX tests =============================================

set(PYTEST_COMMON_ENV
  PJRT_NAMES_AND_LIBRARY_PATHS="mlir_tensorrt:$<TARGET_FILE:mlir-tensorrt-pjrt>"
  JAX_PLATFORMS="mlir_tensorrt,cpu"
  LD_LIBRARY_PATH="${MLIR_TRT_TENSORRT_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  PYTHONPATH="${CMAKE_BINARY_DIR}/python_packages/mlir_tensorrt_compiler:${CMAKE_BINARY_DIR}/python_packages/mlir_tensorrt_runtime:${CMAKE_BINARY_DIR}/python_packages/mlir_tensorrt_tools:${CMAKE_BINARY_DIR}/python_packages/mlir_tensorrt_jax:$ENV{PYTHONPATH}"
  MLIR_TRT_TENSORRT_VERSION="${MLIR_TRT_TENSORRT_VERSION}"
  MLIR_TRT_ENABLE_NCCL="$<BOOL:${MLIR_TRT_ENABLE_NCCL}>"
  ENABLE_ASAN="$<BOOL:${ENABLE_ASAN}>"
  LLVM_ENABLE_ASSERTIONS="$<BOOL:${LLVM_ENABLE_ASSERTIONS}>"
  PJRT_FLAGS="$ENV{PJRT_FLAGS}"
)

# Target for running all JAX numpy tests with pytest
add_custom_target(check-mlir-tensorrt-pjrt-jax-numpy
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTEST_COMMON_ENV}
    LONG_TESTS="True"
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/JAX -v
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${test_depends_}
  COMMENT "Running MLIR-TensorRT JAX numpy tests with pytest"
  USES_TERMINAL
)
set_target_properties(check-mlir-tensorrt-pjrt-jax-numpy PROPERTIES FOLDER "Tests")

# Target for running functional suite tests
add_custom_target(check-mlir-tensorrt-pjrt-functional-suite
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTEST_COMMON_ENV}
    LONG_TESTS="True"
    MODEL_FUNCTIONAL="True"
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/JAX/Models/FunctionalSuite -v
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${test_depends_}
  COMMENT "Running MLIR-TensorRT JAX models FunctionalSuite tests with pytest"
  USES_TERMINAL
)
set_target_properties(check-mlir-tensorrt-pjrt-functional-suite PROPERTIES FOLDER "Tests")

# Target for running benchmark suite tests
add_custom_target(check-mlir-tensorrt-pjrt-benchmark-suite
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTEST_COMMON_ENV}
    PYTEST_BENCHMARK_DIR="${CMAKE_BINARY_DIR}/benchmark_results"
    LONG_TESTS="True"
    MODEL_BENCHMARK="True"
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/JAX/Models/BenchmarkSuite -v
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${test_depends_}
  COMMENT "Running MLIR-TensorRT JAX models BenchmarkSuite tests with pytest"
  USES_TERMINAL
)
set_target_properties(check-mlir-tensorrt-pjrt-benchmark-suite PROPERTIES FOLDER "Tests")


if(TARGET check-all-mlir-tensorrt)
  add_dependencies(check-all-mlir-tensorrt check-mlir-tensorrt-pjrt)
  add_dependencies(check-all-mlir-tensorrt-build-only check-mlir-tensorrt-pjrt-build-only)
endif()
