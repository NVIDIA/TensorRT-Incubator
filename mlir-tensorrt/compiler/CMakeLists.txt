set(MLIR_TENSORRT_COMPILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${MLIR_TENSORRT_ROOT_DIR}/executor/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

