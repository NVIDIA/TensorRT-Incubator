include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${MLIR_TENSORRT_ROOT_DIR}/executor/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)

add_subdirectory(include)
add_subdirectory(lib)

