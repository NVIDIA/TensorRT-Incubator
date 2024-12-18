include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_tensorrt_dialect`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_tensorrt.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(MLIRTensorRTDialectPythonSources)
declare_mlir_python_sources(MLIRTensorRTDialectPythonSources.Dialect
  ADD_TO_PARENT MLIRTensorRTDialectPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRTensorRTDialectPythonSources.Dialect
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt"
  TD_FILE dialects/PythonTensorRTOps.td
  SOURCES
    dialects/tensorrt.py
  DIALECT_NAME tensorrt)

set(PYTHON_SOURCE_DIR "${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(MLIRTensorRTDialectPythonSources.Dialect.tensorrt.PyBind
  MODULE_NAME _tensorrt
  ADD_TO_PARENT MLIRTensorRTDialectPythonSources.Dialect.tensorrt
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectTensorRT.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPITensorRTDialect
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(MLIRTensorRTDialectPythonCAPI
  INSTALL_COMPONENT MLIRTensorRTDialectPythonModules
  INSTALL_DESTINATION python_packages/tensorrt/mlir_tensorrt/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_TENSORRT_DIALECT_BINARY_DIR}/python_packages/tensorrt/mlir_tensorrt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRTensorRTDialectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(MLIRTensorRTDialectPythonModules
  ROOT_PREFIX "${MLIR_TENSORRT_DIALECT_BINARY_DIR}/python_packages/tensorrt/mlir_tensorrt"
  INSTALL_PREFIX "python_packages/tensorrt/mlir_tensorrt"
  DECLARED_SOURCES
    MLIRTensorRTDialectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonSources.Core
  COMMON_CAPI_LINK_LIBS
    MLIRTensorRTDialectPythonCAPI
  )
