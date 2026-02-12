add_mlir_tensorrt_library(MLIRTensorRTStablehloToKernel
  StablehloToKernel.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
  MLIRTensorRTKernelDialect
  StablehloOps
)
