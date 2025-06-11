add_mlir_tensorrt_library(MLIRTensorRTStableHloExtIR
  StablehloInferTensorValueRangeImpl.cpp
  StablehloReifyTypeInterfaceImpl.cpp
  StablehloTensorKindOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRTensorRTInferTensorValueRangeInterface
  MLIRTensorRTInterfaces
  StablehloOps
)
