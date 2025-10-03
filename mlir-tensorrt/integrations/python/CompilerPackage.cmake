# ------------------------------------------------------------------------------
# Instantiate the `mlir_tensorrt_compiler` package.
# ------------------------------------------------------------------------------
# - SRC_DIR: Root source directory of the package
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_compiler/mlir_tensorrt/compiler")
# - OUT_DIR: Where the Python package should be populated.
set(OUT_DIR "${MLIR_TENSORRT_ROOT_BINARY_DIR}/python_packages/mlir_tensorrt_compiler")
# - SETUP_PY: Setup file (relative to SRC_DIR). This will be configured (hard copied)
set(SETUP_PY ../../setup.py)

cmake_path(ABSOLUTE_PATH SETUP_PY NORMALIZE
  BASE_DIRECTORY "${SRC_DIR}"
)

configure_file(
  "${SETUP_PY}"
  "${OUT_DIR}/setup.py"
  @ONLY
    )

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(MLIRTensorRTPythonCompiler.Core
  ADD_TO_PARENT MLIRTensorRTPythonCompiler)
declare_mlir_python_sources(MLIRTensorRTPythonCompiler.CompilerAPI
  ADD_TO_PARENT MLIRTensorRTPythonCompiler)

################################################################################
# Pure python sources and generated code
################################################################################

# All the ROOT_DIR should be set the same to the `compiler` module leaf
# directory where the MLIR libs will be embedded. Source files above this level
# (e.g. packaging files) just need to give relative paths to backtrack to the
# right location. This ensures that all files get installed to the correct
# locations.
declare_mlir_python_sources(MLIRTensorRTPythonCompiler.Core.Python
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.Core
  ROOT_DIR "${SRC_DIR}"
  SOURCES
    ../../README.md
    ../../pyproject.toml
  )

declare_mlir_python_sources(MLIRTensorRTPythonCompiler.CompilerAPI.Python
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.CompilerAPI
  ROOT_DIR "${SRC_DIR}"
  SOURCES
    api.py
    _mlir_libs/_api.pyi
  )

if(MLIR_TRT_ENABLE_TORCH)
  declare_mlir_python_sources(MLIRTensorRTPythonCompiler.TorchBridge
  ADD_TO_PARENT MLIRTensorRTPythonCompiler
  ROOT_DIR "${SRC_DIR}"
  SOURCES
    torch_bridge.py
  )
endif()


################################################################################
# Dialect bindings
################################################################################
foreach(dialect IN LISTS MLIR_TRT_PYTHON_UPSTREAM_DIALECTS_EMBED)
  set_property(TARGET MLIRTensorRTPythonCompiler.Dialects APPEND PROPERTY mlir_python_DEPENDS
    MLIRPythonSources.Dialects.${dialect})
endforeach()

# Add the tensorrt dialect from the 'tensorrt/python' directory.
set_property(TARGET MLIRTensorRTPythonCompiler.Dialects APPEND PROPERTY mlir_python_DEPENDS
  MLIRTensorRTDialectPythonSources.Dialect.tensorrt)

################################################################################
# Python extensions.
################################################################################

# Declare the site initializer.
declare_mlir_python_extension(MLIRTensorRTPythonCompiler.SiteInitializer.PyBind
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT MLIRTensorRTPythonCompiler
  SOURCES
    bindings/Compiler/SiteInitializer.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPIRegisterAllDialects
  PRIVATE_LINK_LIBS
    LLVMSupport
  )

set(COMPILER_OPTIONAL_EMBED_CAPI_LINK_LIBS)
set(COMPILER_OPTIONAL_PRIVATE_LINK_LIBS)
if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND COMPILER_OPTIONAL_EMBED_CAPI_LINK_LIBS
    TensorRTDynamicLoader
    )
  list(APPEND COMPILER_OPTIONAL_PRIVATE_LINK_LIBS
    ${MLIR_TRT_CUDA_TARGET}
    MLIRTensorRTDialectIncludes
    TensorRTHeaderOnly
    )
endif()

# Compiler the compiler Pybind11 module.
declare_mlir_python_extension(MLIRTensorRTPythonCompiler.CompilerAPI.PyBind
  MODULE_NAME _api
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.CompilerAPI
  SOURCES
      bindings/Compiler/CompilerPyBind.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPICommon
    MLIRTensorRTCAPICompiler
    MLIRTensorRTCAPIExecutorTranslations
    MLIRTensorRTCAPISupportStatus
    MLIRCAPITransforms
    ${COMPILER_OPTIONAL_EMBED_CAPI_LINK_LIBS}
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRTensorRTCommonIncludes
    ${COMPILER_OPTIONAL_PRIVATE_LINK_LIBS}
  )


################################################################################
# The fully assembled package of modules.
# This must come last (except for wheel target)
################################################################################

set(source_groups
  MLIRPythonSources.Core
  MLIRTensorRTPythonCompiler)
if(MLIR_TRT_ENABLE_HLO)
  list(APPEND source_groups
    ChloPythonSources
    StablehloPythonSources
    ChloPythonExtensions
    StablehloPythonExtensions
  )
endif()
if(MLIR_TRT_ENABLE_TORCH)
  list(APPEND source_groups
    TorchMLIRPythonSources
    TorchMLIRPythonExtensions
  )
endif()

add_mlir_python_modules("MLIRTensorRTPythonCompilerModules"
  ROOT_PREFIX "${OUT_DIR}/mlir_tensorrt/compiler"
  INSTALL_PREFIX "python_packages/mlir_tensorrt_compiler/mlir_tensorrt/compiler"
  DECLARED_SOURCES ${source_groups}
  COMMON_CAPI_LINK_LIBS
    MTRT
  )
