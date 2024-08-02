add_mlir_tensorrt_library(MLIRTensorRTStableHloExtIR
  StablehloTensorKindOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  StablehloOps
  MLIRDialectUtils
  MLIRIR
  MLIRTensorRTInterfaces
)
