if(MLIR_ENABLE_BINDINGS_PYTHON AND NOT Python3_EXECUTABLE)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

function(add_tensorrt_encoding_def_gen targetName inputFileName outputFileName )
  # Creates `targetName` that invokes tensorrt-tblgen
  # on a [dialect]Ops.td file to generate implementations for the
  # TensorRTEncodingOpInterface's encodeOp interface method.
  list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  COMMAND tensorrt-tblgen --gen-tensorrt-layer-add-defs
    "${inputFileName}"
    -I "${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include"
    ${_mlir_includes}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
  DEPENDS "${inputFileName}" tensorrt-tblgen
  )
  add_custom_target(${targetName} DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
endfunction()


function(add_mtrtd_library name)
  mtrt_add_project_library(${name}
    PROJECT_NAME TensorRTDialect
    ${ARGN}
  )
endfunction()

set(MLIR_TENSORRT_DIALECT_LIB_DIR "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")

set(MLIR_TENSORRT_DIALECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(MLIR_TENSORRT_DIALECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_library(MLIRTensorRTDialectIncludes INTERFACE)
target_include_directories(MLIRTensorRTDialectIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)
mtrt_add_install(MLIRTensorRTDialectIncludes UMBRELLA mtrt-headers)

get_target_property(common_include_dirs
                    MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${common_include_dirs})

include_directories(${MLIR_TENSORRT_DIALECT_SOURCE_DIR}/include)
include_directories(${MLIR_TENSORRT_DIALECT_BINARY_DIR}/include)

add_subdirectory(include/mlir-tensorrt-dialect)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

mtrt_add_header_installation_components(mtrt-headers
  DIRECTORIES
   "include/mlir-tensorrt-dialect"
   "include/mlir-tensorrt-dialect-c"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt-dialect"
   "${CMAKE_CURRENT_BINARY_DIR}/include/mlir-tensorrt-dialect-c"
  )
