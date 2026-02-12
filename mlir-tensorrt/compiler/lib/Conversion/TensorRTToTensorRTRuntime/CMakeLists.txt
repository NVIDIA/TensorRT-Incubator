add_mlir_tensorrt_library(MLIRTensorRTTensorRTToTensorRTRuntime
  TensorRTToTensorRTRuntime.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDAUtils
    MLIRTensorRTExecutorUtils
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTTensorRTRuntimeDialect
    MLIRTensorRTAnalysis
    MLIRTensorRTExecutorPasses
  MLIR_LIBS PUBLIC
    MLIRRewrite
)
