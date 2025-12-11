add_mlir_tensorrt_backend_library(MLIRTensorRTHostBackend
  HostBackend.cpp
  Passes.cpp

  TD
    "mlir-tensorrt/Backends/Host/HostBackend.td"


  LINK_LIBS PUBLIC
  ChloOps
  MLIRTensorRTCommonConvertToLoops
  MLIRTensorRTLinalgElementwiseFusion
  MLIRTensorRTPlanDialect
  MLIRTensorRTSCFDetensorizeLoops
  MLIRTensorRTStableHloExtUtils
  MLIRTensorRTStablehloToLinalg
  MLIRTensorRTTransforms
  StablehloLinalgTransforms
  StablehloOps
  )
