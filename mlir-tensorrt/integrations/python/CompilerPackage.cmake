# ------------------------------------------------------------------------------
# Instantiate the `mlir_tensorrt_compiler` package.
# ------------------------------------------------------------------------------
# - SRC_DIR: Root source directory of the package
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_compiler/mlir_tensorrt/compiler")
# - SETUP_PY: Setup file (relative to SRC_DIR). This will be configured (hard copied)
set(SETUP_PY ../../setup.py)
set(MLIR_TENSORRT_COMPILER_PYTHON_PACKAGE_INSTALL_PREFIX
    "python_packages/mlir_tensorrt_compiler/mlir_tensorrt/compiler")

cmake_path(ABSOLUTE_PATH SETUP_PY NORMALIZE
  BASE_DIRECTORY "${SRC_DIR}"
)

configure_file(
  "${SETUP_PY}"
  "${PYTHON_PACKAGES_BINARY_DIR}/mlir_tensorrt_compiler/setup.py"
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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_compiler/mlir_tensorrt/compiler"
  TD_FILE dialects/PythonTensorRTOps.td
  SOURCES
    dialects/tensorrt.py
  DIALECT_NAME tensorrt)

################################################################################
# Python extensions.
################################################################################

mtrt_add_python_extension(MLIRTensorRTTensorRTDialectPythonExtension
  SOURCES bindings/Compiler/DialectTensorRT.cpp
  EXTENSION_NAME _tensorrt
  PRIVATE_LINK_LIBS
    MLIRTensorRTDialectIncludes
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPITensorRTDialect
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.Dialects
)

# Declare the site initializer.
mtrt_add_python_extension(MLIRTensorRTPythonCompilerSiteInitializerExtension
  SOURCES bindings/Compiler/SiteInitializer.cpp
  EXTENSION_NAME _site_initialize_0
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPIRegisterAllDialects
  ADD_TO_PARENT MLIRTensorRTPythonCompiler
  )

set(COMPILER_OPTIONAL_PRIVATE_LINK_LIBS)
if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND COMPILER_OPTIONAL_PRIVATE_LINK_LIBS
    ${MLIR_TRT_CUDA_TARGET}
    TensorRTHeaderOnly
    )
endif()

# Compiler the compiler Pybind11 module.
mtrt_add_python_extension(MLIRTensorRTPythonCompilerAPIExtension
  SOURCES bindings/Compiler/CompilerPyBind.cpp
  EXTENSION_NAME _api
  ADD_TO_PARENT MLIRTensorRTPythonCompiler.CompilerAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRTensorRTCommonIncludes
    ${COMPILER_OPTIONAL_PRIVATE_LINK_LIBS}
    MLIRTensorRTCAPICommon
  EMBED_CAPI_LINK_LIBS
    MLIRTensorRTCAPIExecutorTranslations
    MLIRTensorRTCAPICompiler
  )

################################################################################
# Aggregate library
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

add_mlir_python_common_capi_library(MLIRTensorRTCompilerPythonCAPI
  INSTALL_COMPONENT MLIRTensorRTCompilerPythonModules
  INSTALL_DESTINATION "${MLIR_TENSORRT_COMPILER_PYTHON_PACKAGE_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_TENSORRT_ROOT_BINARY_DIR}/${MLIR_TENSORRT_COMPILER_PYTHON_PACKAGE_INSTALL_PREFIX}/_mlir_libs"
  DECLARED_SOURCES
    ${source_groups}
  )

################################################################################
# The fully assembled package of modules.
# This must come last (except for wheel target)
################################################################################

add_mlir_python_modules(MLIRTensorRTCompilerPythonModules
  ROOT_PREFIX "${MLIR_TENSORRT_ROOT_BINARY_DIR}/${MLIR_TENSORRT_COMPILER_PYTHON_PACKAGE_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_TENSORRT_COMPILER_PYTHON_PACKAGE_INSTALL_PREFIX}"
  DECLARED_SOURCES ${source_groups}
  COMMON_CAPI_LINK_LIBS
    MLIRTensorRTCompilerPythonCAPI
    MLIRCAPITransforms
  )

################################################################################
# Fixup nanobind compilation options.
################################################################################
foreach(nb_target nanobind-static nanobind-static-ft)
  if(TARGET ${nb_target})
    target_compile_options(${nb_target} PUBLIC -Wno-cast-qual -Wno-zero-length-array -Wno-nested-anon-types -Wno-c++98-compat-extra-semi -Wno-covered-switch-default)
  endif()
endforeach()
