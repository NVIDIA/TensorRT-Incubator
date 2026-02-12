add_mlir_tensorrt_library(MLIRTensorRTStablehloToLinalg
  StablehloToLinalg.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    StablehloLinalgTransforms
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRLinalgTransforms
  )
