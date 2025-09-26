add_mlir_tensorrt_library(MLIRTensorRTStableHloExtIR
  StablehloInferTensorValueRangeImpl.cpp
  StablehloReifyTypeInterfaceImpl.cpp
  StablehloTensorKindOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTInferTensorValueRangeInterface
    MLIRTensorRTCommonTensorKindInterface
    StablehloOps
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRTensorDialect
)
