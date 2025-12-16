add_mlir_tensorrt_library(
  MLIRTensorRTStableHloToExecutableKernelGenExtension
  KernelGenExtension.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerClient
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTCompilerStableHloToExecutable
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
