add_mlir_tensorrt_library(MLIRTensorRTStablehloToKernel
  StablehloToKernel.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTensorRTKernelDialect
  StablehloOps
)

