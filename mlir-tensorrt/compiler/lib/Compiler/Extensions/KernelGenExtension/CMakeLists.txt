add_mlir_tensorrt_library(
  MLIRTensorRTCompilerKernelGenExtension
  KernelGenExtension.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerClient
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTStablehloInputPipeline
    MLIRTensorRTExecuteConstantFoldableSubgraphs
    MLIRTensorRTKernelBackend
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelPipelines
    MLIRTensorRTKernelToCUDA
    MLIRTensorRTKernelTransforms
    MLIRTensorRTKernelTransformSchedules
    MLIRTensorRTLinalgElementwiseFusion
    MLIRTensorRTLinalgSimplifyExtractSlice
    MLIRTensorRTStablehloToKernel
    MLIRTensorRTStablehloToLinalg
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRTensorToLinalg
  )
