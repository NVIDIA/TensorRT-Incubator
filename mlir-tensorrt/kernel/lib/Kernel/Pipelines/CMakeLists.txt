add_mlir_kernel_library(MLIRTensorRTKernelPipelines
  Pipelines.cpp
  GPUModuleLoweringImpl.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTComplexToStdExt
    MLIRTensorRTKernelTransforms
    MLIRTensorRTKernelTransformSchedules
  MLIR_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRPass
  )
