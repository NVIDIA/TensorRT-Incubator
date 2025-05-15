add_mlir_tensorrt_library(MLIRTensorRTStableHloExtIR
  StablehloTensorKindOpInterfaceImpl.cpp
  StableHloReifyTypeInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRTensorRTInterfaces
  StablehloOps
)
