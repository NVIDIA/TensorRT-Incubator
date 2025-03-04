set(MLIR_TENSORRT_COMPILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")


set(MLIR_TENSORRT_COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PARENT_SCOPE)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

