add_mlir_tensorrt_library(MLIRTensorRTUtilsShapeInfo
  ShapeInfo.cpp
  ModuleUtils.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRDialectUtils
  MLIRFuncDialect
  )
