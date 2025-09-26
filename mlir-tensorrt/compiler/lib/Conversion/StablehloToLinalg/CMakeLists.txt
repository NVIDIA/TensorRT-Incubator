add_mlir_tensorrt_library(MLIRTensorRTStablehloToLinalg
  StablehloToLinalg.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    StablehloLinalgTransforms
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRLinalgTransforms
  )
