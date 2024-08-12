# ------------------------------------------------------------------------------
# Declare bindings for upstream packages.
# ------------------------------------------------------------------------------


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
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheels/${name}")
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
