add_mlir_tensorrt_backend_library(MLIRTensorRTKernelBackend
  KernelBackend.cpp
  KernelSegmentation.cpp

  TD
    "mlir-tensorrt/Backends/Kernel/KernelBackend.td"

  LINK_LIBS PUBLIC
  ChloOps
  MLIRTensorRTCompilerClient
  MLIRTensorRTHostBackend
  MLIRTensorRTKernelDialect
  MLIRTensorRTPlanDialect
  MLIRTensorRTStableHloExtUtils
  StablehloOps
  )
