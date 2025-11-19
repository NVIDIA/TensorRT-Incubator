set(PYTHON_PACKAGES_BINARY_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages")

if(MLIR_TRT_ENABLE_PYTHON)
  declare_mlir_python_sources(MLIRTensorRTPythonCompiler)
  declare_mlir_python_sources(MLIRTensorRTPythonCompiler.Dialects
    ADD_TO_PARENT MLIRTensorRTPythonCompiler)

  find_package(DLPack REQUIRED)
endif()


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
  ${MLIR_TENSORRT_ROOT_DIR}/tensorrt/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/compiler/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/kernel/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/tensorrt/include)

if(MLIR_TRT_ENABLE_HLO)
  include_directories(${Stablehlo_SOURCE_DIR})
  if(NOT EXISTS "${Stablehlo_SOURCE_DIR}")
    message(FATAL_ERROR "Stablehlo_SOURCE_DIR is incorrect")
  endif()
endif()

if(MLIR_TRT_ENABLE_TORCH)
  if(NOT EXISTS "${torch_mlir_SOURCE_DIR}/include")
    message(FATAL_ERROR "torch_mlir_SOURCE_DIR is incorrect")
  endif()
  include_directories(${torch_mlir_SOURCE_DIR}/include)
endif()

if(MLIR_TRT_ENABLE_PYTHON)
  # This is the directory under `llvm-project/mlir/python`.
  set(MLIR_MAIN_PYTHON_DIR "${MLIR_MAIN_SRC_DIR}/python")
  # This is the C++ source directory under `llvm-project/mlir/lib/Bindings/Python`.
  set(MLIR_MAIN_PYTHON_BINDINGS_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")

  # -------------------------------------------------
  # mlir_tensorrt_compiler package
  # -------------------------------------------------

  include(CompilerPackage.cmake)

  # -------------------------------------------------
  # mlir_tensorrt_runtime package
  # -------------------------------------------------
  configure_file(
    mlir_tensorrt_runtime/setup.py
    ${PYTHON_PACKAGES_BINARY_DIR}/mlir_tensorrt_runtime/setup.py
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

  mtrt_add_python_extension(MLIRTensorRTRuntimePythonExtension
    bindings/Runtime/RuntimePyBind.cpp
    ROOT_DIR "${PYTHON_PACKAGES_BINARY_DIR}"
    OUTPUT_DIR
      "mlir_tensorrt_runtime/mlir_tensorrt/runtime/_mlir_libs"
    EXTENSION_NAME _api
    PRIVATE_LINK_LIBS
      MLIRTensorRTCAPISupportStatus
      MLIRTensorRTCAPICommon
      MLIRTensorRTCAPIRuntime
      LLVMSupport
      DLPack::Headers
    )

  add_mlir_python_modules(MLIRTensorRTPythonRuntimeModules
    ROOT_PREFIX "${PYTHON_PACKAGES_BINARY_DIR}/mlir_tensorrt_runtime/mlir_tensorrt/runtime"
    INSTALL_PREFIX "python_packages/mlir_tensorrt_runtime/mlir_tensorrt/runtime"
    DECLARED_SOURCES
      MLIRTensorRTPythonRuntimeSources
    )
  add_dependencies(MLIRTensorRTPythonRuntimeModules MLIRTensorRTRuntimePythonExtension)

  # -------------------------------------------------
  # mlir_tensorrt_tools package
  # -------------------------------------------------
  configure_file(
    mlir_tensorrt_tools/setup.py
    ${PYTHON_PACKAGES_BINARY_DIR}/mlir_tensorrt_tools/setup.py
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
    ROOT_PREFIX "${PYTHON_PACKAGES_BINARY_DIR}/mlir_tensorrt_tools"
    INSTALL_PREFIX "python_packages/mlir_tensorrt_tools"
    DECLARED_SOURCES
      MLIRTensorRTPythonToolsSources
    )
endif()
