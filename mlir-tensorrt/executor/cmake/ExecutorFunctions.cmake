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
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC}" flatc
  )
  add_custom_target(${target}
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${generatedFileName}"
  )
endfunction()

# --------------------------------------------------------------
# Add target to install set
# --------------------------------------------------------------
function(add_mlir_executor_install target)
  install(TARGETS ${target}
    LIBRARY
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_Executor_Runtime
    ARCHIVE
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_Executor_Development
    RUNTIME
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT MTRT_Executor_Runtime
    OBJECTS
      DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT MTRT_Executor_Development
  )
endfunction()

# --------------------------------------------------------------
# Add includes to executor library
# --------------------------------------------------------------
function(populate_mlir_executor_includes_helper name relationship)
  target_include_directories(${name} ${relationship}
    $<BUILD_INTERFACE:${MLIR_EXECUTOR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MLIR_EXECUTOR_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)
endfunction()
function(populate_mlir_executor_includes name)
  get_target_property(type ${name} TYPE)
  if (${type} STREQUAL "INTERFACE_LIBRARY")
    populate_mlir_executor_includes_helper(${name}
      INTERFACE)
  else()
    populate_mlir_executor_includes_helper(${name}
      PUBLIC)
  endif()
  if(TARGET obj.${name})
    populate_mlir_executor_includes_helper(obj.${name}
      PUBLIC)
  endif()
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_library`
# --------------------------------------------------------------
function(add_mlir_executor_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXECUTOR_LIBS ${name})
  add_mlir_library(${name} DISABLE_INSTALL ${ARGN})
  populate_mlir_executor_includes(${name})
  add_mlir_executor_install(${name})
endfunction()

# --------------------------------------------------------------
# Wrapper around `add_mlir_library`
# --------------------------------------------------------------
function(add_mlir_executor_runtime_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXECUTOR_RUNTIME_LIBS ${name})
  add_mlir_library(${name} ${ARGN})
  populate_mlir_executor_includes(${name})
  add_mlir_executor_install(${name})
endfunction()
