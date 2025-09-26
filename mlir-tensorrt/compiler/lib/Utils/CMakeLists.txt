add_mlir_tensorrt_library(MLIRTensorRTUtilsShapeInfo
  ShapeInfo.cpp
  ModuleUtils.cpp

  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRDialectUtils
    MLIRFuncDialect
  )
