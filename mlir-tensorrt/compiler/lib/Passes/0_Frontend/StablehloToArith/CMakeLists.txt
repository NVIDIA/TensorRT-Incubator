add_mlir_tensorrt_library(MLIRTensorRTStablehloToArith
  StablehloToArith.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
)
