string(REGEX MATCH "([0-9]+)\.([0-9]+)" TRT_VERSION "${MLIR_TRT_TENSORRT_VERSION}")
string(REPLACE "." "" TRT_VERSION "${TRT_VERSION}")
set(PKG_VERSION "${MLIR_TENSORRT_VERSION}+cuda12.trt${TRT_VERSION}")
set(WHEEL_OUTPUT_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/wheels/python${Python3_VERSION}/trt${TRT_VERSION}")

# Specifies that all MLIR packages are co-located under the `mlir_tensorrt.compiler`
# top level package. This definition is used in the MLIR Python binding headers for
# generating lookup paths when the MLIR pybind modules are loaded at runtime.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_tensorrt.compiler.")

# LLVM `mlir_tablegen` function depends on having include directories
# declared in the directory scopped include directories property.
include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${MLIR_TENSORRT_ROOT_DIR}/compiler/include
  ${MLIR_TENSORRT_ROOT_DIR}/kernel/include
  ${MLIR_TENSORRT_ROOT_DIR}/executor/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/compiler/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/kernel/include
  # Upstream stablehlo python extension modules doesn't properly declare include
  # requirements.
  "$<$<BOOL:${MLIR_TRT_ENABLE_HLO}>:${stablehlo_SOURCE_DIR}>"
  )

# Create an aggregate target for building all wheels. The below
# `add_mtrt_python_wheel` calls declare targets that can be built independently, but
# they are also added to this target
add_custom_target(mlir-tensorrt-all-wheels)

include(CompilerPackageUtils.cmake)

find_package(CUDAToolkit REQUIRED)

# -------------------------------------------------
# mlir_tensorrt_compiler package
# -------------------------------------------------
if(MLIR_TRT_ENABLE_PYTHON)
  # This is the directory under `llvm-project/mlir/python`.
  set(MLIR_MAIN_PYTHON_DIR "${MLIR_MAIN_SRC_DIR}/python")
  # This is the C++ source directory under `llvm-project/mlir/lib/Bindings/Python`.
  set(MLIR_MAIN_PYTHON_BINDINGS_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")

  mlir_tensorrt_declare_compiler_python_package(MLIRTensorRTPythonCompiler
    SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_compiler/mlir_tensorrt/compiler"
    OUT_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_compiler"
    SETUP_PY ../../setup.py
    WHEEL_TARGET_NAME "mlir-tensorrt-compiler-wheel"
    WHEEL_OUTPUT_DIR ${WHEEL_OUTPUT_DIR}
    MLIR_PYTHON_DIR "${MLIR_MAIN_PYTHON_DIR}"
    MLIR_PYBIND_DIR "${MLIR_MAIN_PYTHON_BINDINGS_DIR}"
    VERSION ${PKG_VERSION}
    SOURCES
      ../../README.md
      ../../pyproject.toml
      api.py
      _mlir_libs/_api.pyi
  )

  # -------------------------------------------------
  # mlir_tensorrt_runtime package
  # -------------------------------------------------
  configure_file(
    mlir_tensorrt_runtime/setup.py
    ${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_runtime/setup.py
    @ONLY
    )

  declare_mlir_python_sources(MLIRTensorRTPythonRuntimeSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_runtime/mlir_tensorrt/runtime"
    SOURCES
      ../../README.md
      ../../pyproject.toml
      _mlir_libs/_api.pyi
      api.py
    )

  declare_mlir_python_extension(MLIRTensorRTPythonRuntimeSources.Runtime.PyBind
    MODULE_NAME _api
    ADD_TO_PARENT MLIRTensorRTPythonRuntimeSources
    SOURCES
      bindings/Runtime/RuntimePyBind.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
      MLIRTensorRTCAPISupportStatus
      MLIRTensorRTCAPICommon
      MLIRTensorRTCAPIRuntime
      DLPack::Headers
    )

  add_mlir_python_modules(MLIRTensorRTPythonRuntimeModules
    ROOT_PREFIX "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_runtime/mlir_tensorrt/runtime"
    INSTALL_PREFIX "python_packages/mlir_tensorrt_runtime/mlir_tensorrt/runtime"
    DECLARED_SOURCES
      MLIRTensorRTPythonRuntimeSources
    COMMON_CAPI_LINK_LIBS
    )

  add_mtrt_python_wheel(mlir-tensorrt-runtime-wheel
    DEPENDS MLIRTensorRTPythonRuntimeModules
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_runtime"
    PACKAGE_BINARY_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_runtime"
    OUTPUT_DIR "${WHEEL_OUTPUT_DIR}"
    ADD_TO_PARENT mlir-tensorrt-all-wheels
    VERSION "${PKG_VERSION}"
    )

  # -------------------------------------------------
  # mlir_tensorrt_tools package
  # -------------------------------------------------
  configure_file(
    mlir_tensorrt_tools/setup.py
    ${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_tools/setup.py
    @ONLY
    )

  declare_mlir_python_sources(MLIRTensorRTPythonToolsSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_tools"
    SOURCES
      README.md
      pyproject.toml
      mlir_tensorrt/tools/__init__.py
      mlir_tensorrt/tools/gpu_tools.py
  )

  add_mlir_python_modules(MLIRTensorRTPythonToolsModules
    ROOT_PREFIX "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_tools"
    INSTALL_PREFIX "python_packages/mlir_tensorrt_tools"
    DECLARED_SOURCES
      MLIRTensorRTPythonToolsSources
    )

  add_mtrt_python_wheel(mlir-tensorrt-tools-wheel
    DEPENDS MLIRTensorRTPythonToolsModules
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_tools"
    PACKAGE_BINARY_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_tools"
    OUTPUT_DIR "${WHEEL_OUTPUT_DIR}"
    ADD_TO_PARENT mlir-tensorrt-all-wheels
    VERSION "${PKG_VERSION}"
    )
endif()
