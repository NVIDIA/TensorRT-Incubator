set(MLIR_TENSORRT_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_TENSORRT_COMMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(MLIRTensorRTCommonIncludes INTERFACE)
target_include_directories(MLIRTensorRTCommonIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

add_subdirectory(tools/mtrt-tblgen)
add_subdirectory(include/mlir-tensorrt-common)
add_subdirectory(lib)

add_dependencies(MLIRTensorRTCommonIncludes MLIRTensorRTStatusEnumIncGen)

install(TARGETS MLIRTensorRTCommonIncludes
  EXPORT MLIRTensorRTCommonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
