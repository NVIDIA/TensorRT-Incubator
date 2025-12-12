add_mlir_kernel_library(MLIRTensorRTKernelTransformOps
  KernelTransformOps.cpp

  DEPENDS
  MLIRTensorRTKernelTransformOpsIncGen

  LINK_LIBS PUBLIC
    MLIRKernelOutliningUtils
    MLIRKernelTensorUtils
    MLIRTensorRTCommonToLoopsOpInterface
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelTransforms
    MLIRTensorRTRegionUtils
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRBufferizationDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRNVGPUDialect
    MLIRNVVMDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorTransforms
    MLIRTransformDialect
    MLIRTransforms
    MLIRVectorDialect
)
