# ------------------------------------------------------------------------------
# Declare bindings for upstream packages.
# ------------------------------------------------------------------------------
function(populate_upstream_mlir_python_binding_dependencies
    MLIR_PYTHON_DIR
    MLIR_PYBIND_DIR
    parent_source_group)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/FuncOps.td
    SOURCES
      dialects/func.py
    DIALECT_NAME func)

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/MathOps.td
    SOURCES
      dialects/math.py
    DIALECT_NAME math)

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/ComplexOps.td
    SOURCES
      dialects/complex.py
    DIALECT_NAME complex)

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/ArithOps.td
    SOURCES
      dialects/arith.py
    DIALECT_NAME arith
    GEN_ENUM_BINDINGS)

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/AffineOps.td
    SOURCES
      dialects/affine.py
    DIALECT_NAME affine
    GEN_ENUM_BINDINGS)

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/TensorOps.td
    SOURCES
      dialects/tensor.py
    DIALECT_NAME tensor)

  declare_mlir_python_sources(${parent_source_group}.quant
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    GEN_ENUM_BINDINGS
    SOURCES
      dialects/quant.py
      _mlir_libs/_mlir/dialects/quant.pyi
  )

  declare_mlir_python_extension(${parent_source_group}.Quant.Pybind
    MODULE_NAME _mlirDialectsQuant
    ADD_TO_PARENT ${parent_source_group}.quant
    ROOT_DIR "${ARG_MLIR_PYBIND_DIR}"
    SOURCES
      DialectQuant.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      MLIRCAPIIR
      MLIRCAPIQuant
  )

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/BufferizationOps.td
    SOURCES
      dialects/bufferization.py
    DIALECT_NAME bufferization
    GEN_ENUM_BINDINGS_TD_FILE
      "../../include/mlir/Dialect/Bufferization/IR/BufferizationEnums.td"
  )

  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ${parent_source_group}
    ROOT_DIR "${ARG_MLIR_PYTHON_DIR}/mlir"
    TD_FILE dialects/SCFOps.td
    SOURCES
      dialects/scf.py
    DIALECT_NAME scf
  )
endfunction()

# ------------------------------------------------------------------------------
# Instantiate a `mlir_tensorrt_compiler` package.
# Args:
# - SRC_DIR: Root source directory of the package
# - OUT_DIR: Where the Python package should be populated.
# - SETUP_PY: Setup file (relative to SRC_DIR). This will be configured (hard copied)
#   using @ONLY directly into the OUT_DIR.
# - VERSION: The package version (including any features suffixes like '+cuda12')
# - SOURCES: List of sources in the package (exluding setup.py), relative from
#    'SRC_DIR'
# - MLIR_PYTHON_DIR: The directory under `llvm-project/mlir/python`.
# - MLIR_PYTHON_BINDINGS_DIR: The the C++ source directory under
#   `llvm-project/mlir/lib/Bindings/Python`.
# ------------------------------------------------------------------------------
function(mlir_tensorrt_declare_compiler_python_package pkg_group)
  cmake_parse_arguments(ARG ""
    "SRC_DIR;OUT_DIR;SETUP_PY;WHEEL_TARGET_NAME;WHEEL_OUTPUT_DIR;VERSION;MLIR_PYTHON_DIR;MLIR_PYBIND_DIR"
    "SOURCES" ${ARGN})

  cmake_path(ABSOLUTE_PATH ARG_SETUP_PY NORMALIZE
    BASE_DIRECTORY "${ARG_SRC_DIR}"
  )
  configure_file(
    "${ARG_SETUP_PY}"
    "${ARG_OUT_DIR}/setup.py"
    @ONLY
      )

  # All the ROOT_DIR should be set the same to the `compiler` module leaf
  # directory where the MLIR libs will be embedded. Source files above this level
  # (e.g. packaging files) just need to give relative paths to backtrack to the
  # right location. This ensures that all files get installed to the correct
  # locations.
  declare_mlir_python_sources(${pkg_group}
    ROOT_DIR "${ARG_SRC_DIR}"
    SOURCES ${ARG_SOURCES}
  )

  declare_mlir_python_sources("${pkg_group}.Dialects"
    ADD_TO_PARENT ${pkg_group}
  )

  declare_mlir_dialect_python_bindings(
    DIALECT_NAME tensorrt
    ADD_TO_PARENT ${pkg_group}.Dialects
    ROOT_DIR "${ARG_SRC_DIR}"
    TD_FILE
      dialects/TensorRTOps.td
    SOURCES
      dialects/tensorrt.py
  )

  declare_mlir_python_extension(${pkg_group}.Dialects.tensorrt.PyBind
    MODULE_NAME _tensorrt
    ADD_TO_PARENT ${pkg_group}.Dialects.tensorrt
    SOURCES
      bindings/Compiler/Dialects/DialectTensorRT.cpp
    EMBED_CAPI_LINK_LIBS
      MLIRTensorRTCAPITensorRTDialect
    PRIVATE_LINK_LIBS
      LLVMSupport
  )

  set(site_initializer_src_ bindings/Compiler/SiteInitializer.cpp)
  set(site_initializer_link_libs_ MLIRTensorRTCAPIRegisterAllDialects)
  declare_mlir_python_extension(${pkg_group}.SiteInitializer.PyBind
    MODULE_NAME _site_initialize_0
    ADD_TO_PARENT ${pkg_group}
    SOURCES
      ${site_initializer_src_}
    EMBED_CAPI_LINK_LIBS
      ${site_initializer_link_libs_}
    PRIVATE_LINK_LIBS
      LLVMSupport
  )

  populate_upstream_mlir_python_binding_dependencies(
    "${ARG_MLIR_PYTHON_DIR}"
    "${ARG_MLIR_PYBIND_DIR}"
    "${pkg_group}.Dialects")

  declare_mlir_python_extension(${pkg_group}.Compiler.PyBind
    MODULE_NAME _api
    ADD_TO_PARENT ${pkg_group}
    SOURCES
        bindings/Compiler/CompilerPyBind.cpp
    EMBED_CAPI_LINK_LIBS
      MLIRTensorRTCAPICompiler
      MLIRTensorRTCAPISupportStatus
      MLIRTensorRTCAPICommon
      MLIRCAPITransforms
    PRIVATE_LINK_LIBS
      LLVMSupport
  )

  set(_deps_sources
    MLIRPythonSources.Core
    ${pkg_group})
  if(MLIR_TRT_ENABLE_HLO)
    list(APPEND _deps_sources
      ChloPythonSources
      StablehloPythonSources
      ChloPythonExtensions
      StablehloPythonExtensions
    )
  endif()

  add_mlir_python_common_capi_library("${pkg_group}CLibrary"
    INSTALL_COMPONENT "${pkg_group}Modules"
    INSTALL_DESTINATION python_packages/mlir_tensorrt_compiler/mlir_tensorrt/compiler/_mlir_libs
    OUTPUT_DIRECTORY "${ARG_OUT_DIR}/mlir_tensorrt/compiler/_mlir_libs"
    DECLARED_SOURCES
    ${_deps_sources}
  )

  add_mlir_python_modules("${pkg_group}Modules"
    ROOT_PREFIX "${ARG_OUT_DIR}/mlir_tensorrt/compiler"
    INSTALL_PREFIX "python_packages/mlir_tensorrt_compiler/mlir_tensorrt/compiler"
    DECLARED_SOURCES
    ${_deps_sources}
    COMMON_CAPI_LINK_LIBS
    "${pkg_group}CLibrary"
  )

  # Add compiler defs and required libraries to the compiler pybind11 api module.
  # The name of the library is created programatically by MLIR's cmake utilities,
  # which is why it looks a bit strange here.
  _mtrt_set_target_compile_defs(${pkg_group}Modules.extension._api.dso)
  target_link_libraries(
    ${pkg_group}Modules.extension._api.dso
    PUBLIC
    TensorRTHeaderOnly
    MLIRTRTTensorRTDynamicLoader
    CUDA::cudart
  )

  if(ARG_WHEEL_TARGET_NAME)
    add_mtrt_python_wheel(${ARG_WHEEL_TARGET_NAME}
      DEPENDS
      "${pkg_group}Modules"
      PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tensorrt_compiler"
      PACKAGE_BINARY_DIR "${ARG_OUT_DIR}"
      OUTPUT_DIR "${ARG_WHEEL_OUTPUT_DIR}"
      ADD_TO_PARENT mlir-tensorrt-all-wheels
      VERSION "${ARG_VERSION}"
    )
  endif()
endfunction()


# ------------------------------------------------------------------------------
# Extracts the name for a python package by reading the given pyproject.toml
# file and extracting the metadata.
# ------------------------------------------------------------------------------
function(_mtrt_extract_pyproject_metadata filename name_outvar)
  _mtrt_find_in_file("${filename}"
    [[^ *name *= *"(.*)" *$]]
    "\\1"
    "${name_outvar}"
    )
  return(PROPAGATE "${name_outvar}")
endfunction()

# ------------------------------------------------------------------------------
# Calculate the expected python wheel filename using the filename given by
# by the Python package binary format specification. This can then be used
# to check that a built wheel exists with the specified name.
# ------------------------------------------------------------------------------
function(_mtrt_get_expected_wheel_name out_var)
  cmake_parse_arguments(ARG "" "NAME;VERSION" "" ${ARGN})
  # From: https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-format
  # "The wheel filename is {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl."
  # The Python tag indicates the implementation required by package.
  set(PY_TAG "cp${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
  # The ABI tag indicates which Python ABI is required by included extension modules.
  set(ABI_TAG "cp${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
  # "The platform tag is simply `distutils.util.get_platform()` with all hyphens and
  # periods replaced by underscore."
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_platform())"
    OUTPUT_VARIABLE PLATFORM_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX REPLACE "[\.\-]" "_" "${PLATFORM_TAG}" PLATFORM_TAG)
  set("${out_var}"
    "${ARG_NAME}-${ARG_VERSION}-${PY_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl"
    PARENT_SCOPE
    )
endfunction()

# ------------------------------------------------------------------------------
# Creates `name` target that generates a Python .whl file suitable for
# distribution. This command is only valid when BUILD_SHARED_LIBS is false
# (otherwise the wheel needs to be built from the install tree).
# ------------------------------------------------------------------------------
function(add_mtrt_python_wheel name)
  if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR
      "Python wheels cannot be built from build tree when BUILD_SHARED_LIBS=ON! "
      "This must be done after build from an install tree and is currently unsupported."
    )
  endif()

  find_package(Python3 REQUIRED)

  cmake_parse_arguments(ARG ""
    "PACKAGE_DIR;OUTPUT_DIR;ADD_TO_PARENT;PACKAGE_BINARY_DIR;VERSION"
    "DEPENDS" ${ARGN})

  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheels/python${Python3_VERSION}/${name}")
  else()
    # Always force the wheel to be generated into a subdirectory.
    set(ARG_OUTPUT_DIR "${ARG_OUTPUT_DIR}/${name}")
  endif()

  _mtrt_extract_pyproject_metadata(
    "${ARG_PACKAGE_DIR}/pyproject.toml"
    PKG_NAME)
  _mtrt_get_expected_wheel_name(wheel_name
    NAME "${PKG_NAME}"
    VERSION "${ARG_VERSION}"
  )
  set(expected_output_path "${ARG_OUTPUT_DIR}/${wheel_name}")
  message(STATUS "Creating python wheel target ${name} (name=${PKG_NAME}, VERSION=${ARG_VERSION}). "
                 "Expected output path = ${expected_output_path}")

  add_custom_command(OUTPUT "${expected_output_path}"
    COMMAND cmake -E remove_directory "${ARG_OUTPUT_DIR}"
    COMMAND cmake -E make_directory "${ARG_OUTPUT_DIR}"
    COMMAND "${Python3_EXECUTABLE}" "-m" "build" "--wheel"
            "--outdir" "${ARG_OUTPUT_DIR}"
    WORKING_DIRECTORY "${ARG_PACKAGE_BINARY_DIR}"
    DEPENDS ${ARG_DEPENDS}
  )
  add_custom_target(${name}
    DEPENDS "${expected_output_path}"
  )

  if(ARG_ADD_TO_PARENT)
    add_dependencies("${ARG_ADD_TO_PARENT}" ${name})
  endif()
endfunction()
