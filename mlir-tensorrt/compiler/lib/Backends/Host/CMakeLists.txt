add_mlir_tensorrt_backend_library(MLIRTensorRTHostBackend
  HostBackend.cpp

  TD
    "mlir-tensorrt/Backends/Host/HostBackend.td"

  LINK_LIBS PUBLIC
  ChloOps
  MLIRTensorRTPlanDialect
  MLIRTensorRTStableHloExtUtils
  StablehloLinalgTransforms
  StablehloOps
  )