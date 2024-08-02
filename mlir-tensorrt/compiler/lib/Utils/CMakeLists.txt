include_directories(${MLIR_TENSORRT_ROOT_DIR}/include)

add_mlir_tensorrt_library(MLIRTensorRTCommonUtils
  PARTIAL_SOURCES_INTENDED
  RegionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRPass
  MLIRSupport
)
