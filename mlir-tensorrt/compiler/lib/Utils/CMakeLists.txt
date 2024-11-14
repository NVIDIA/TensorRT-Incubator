
add_mlir_tensorrt_library(MLIRTensorRTUtilsShapeInfo
  ShapeInfo.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  )
