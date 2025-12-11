add_mlir_tensorrt_backend_library(MLIRTensorRTTensorRTBackend
  TensorRTBackend.cpp
  Passes.cpp

  TD
    "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.td"


  LINK_LIBS PUBLIC
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTHostBackend
  MLIRTensorRTPlanDialect
  # TODO: remove unconditional dependence on Stablehlo dialect.
  MLIRTensorRTStablehloToTensorRT
  StablehloOps
  )
