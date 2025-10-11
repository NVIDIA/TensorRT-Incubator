if(MLIR_ENABLE_BINDINGS_PYTHON AND NOT Python3_EXECUTABLE)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(cmake/TensorRTFunctions.cmake)

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
