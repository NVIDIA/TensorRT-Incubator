add_mlir_tensorrt_library(MLIRTensorRTChloToStablehloExt
  ChloToStablehloExt.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    ChloOps
    MLIRTensorRTTensorRTDialect
    StablehloOps
    StablehloPasses
  MLIR_LIBS PUBLIC
    MLIRDialectUtils
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
  )
