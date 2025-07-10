add_mlir_tensorrt_library(MLIRTensorRTCommonLinalgExtTransforms
  ToLoopsOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorRTCommonIncludes
  MLIRTensorRTCommonToLoopsOpInterface
  )
