add_mlir_tensorrt_library(MLIRTensorRTCommonLinalgExtTransforms
  ToLoopsOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
    MLIRTensorRTCommonToLoopsOpInterface
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRSCFDialect
    MLIRTensorDialect
  )
