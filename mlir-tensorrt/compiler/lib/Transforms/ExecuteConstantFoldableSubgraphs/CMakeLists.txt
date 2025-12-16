add_mlir_tensorrt_library(MLIRTensorRTExecuteConstantFoldableSubgraphs
    ExecuteConstantFoldableSubgraphs.cpp

    DEPENDS
    MLIRTensorRTGenericTransformPassIncGen

    LINK_LIBS PUBLIC
      MLIRTensorRTCompilerClient
      MLIRTensorRTCompilerStableHloToExecutable
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
      MLIRTensorRTExecutorRuntimeAPI
      MLIRTensorRTExecutionEngineLuaRuntime
      MLIRTensorRTTensorExtPadToInsertSlice
    MLIR_LIBS PUBLIC
      MLIRFuncDialect
      MLIRFuncInlinerExtension
      MLIRTensorToLinalg
    )
