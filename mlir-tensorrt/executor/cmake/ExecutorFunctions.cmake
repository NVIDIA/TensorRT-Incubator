# --------------------------------------------------------------
# Creates `targetName` that invokes executor-tblgen
# on a `.td` file to generate enum definitions and
# associated utility functions.
# --------------------------------------------------------------
function(add_mlir_executor_enum_gen targetName inputFileName outputFileName)
  cmake_parse_arguments(ARG "GEN_C" "" "" ${ARGN})

  set(command "--gen-custom-enum-defs")
  if(ARG_GEN_C)
    set(command "--gen-custom-enum-c-defs")
  endif()
  list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  COMMAND executor-tblgen ${command}
    "${CMAKE_CURRENT_LIST_DIR}/${inputFileName}"
    -I "${MLIR_TENSORRT_ROOT_DIR}/include"
    ${_mlir_includes}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  DEPENDS "${inputFileName}" executor-tblgen
  )
  add_custom_target(${targetName} DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
endfunction()

# --------------------------------------------------------------
# Creates `target` that invokes the `flatc` compiler on the
# given SRC, which sould be a flatbuffer schema file.
# It creates target which at build time generates a corresponding
# [filename]Generated.h in the build directory corresponding to the
# source directory of the SRC file.
# --------------------------------------------------------------
function(add_mlir_executor_flatbuffer_schema target)
  set(prefix ARG)
  set(noValues "")
  set(singleValues "SRC")
  cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}"
                        "${multiValues}" ${ARGN})
  get_filename_component(srcFileName "${ARG_SRC}" NAME_WE)
  set(generatedFileName "${srcFileName}Flatbuffer.h")
  add_custom_command(
    OUTPUT "${generatedFileName}"
    COMMAND flatc --cpp --cpp-std c++17 -o ${CMAKE_CURRENT_BINARY_DIR}
            --filename-suffix Flatbuffer "${ARG_SRC}"
            --gen-object-api
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC}"
  )
  add_custom_target(${target}
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${generatedFileName}"
  )
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_library`
# --------------------------------------------------------------
function(add_mlir_executor_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXECUTOR_LIBS ${name})
  add_mlir_library(${name} ${ARGN})
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_library`
# --------------------------------------------------------------
function(add_mlir_executor_runtime_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXECUTOR_RUNTIME_LIBS ${name})
  add_mlir_library(${name} ${ARGN})
endfunction()
