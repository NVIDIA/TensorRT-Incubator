add_mlir_tensorrt_library(MLIRTensorRTHostToEmitC
  HostToEmitC.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTToEmitC
    MLIRTensorRTExecutorDialect
    MLIRTensorRTCUDADialect
    MLIRTensorRTTensorRTRuntimeDialect
    MLIRTensorRTLLVMConversionUtils
  MLIR_LIBS PUBLIC
    MLIRArithToEmitC
    MLIRArithTransforms
    MLIREmitCDialect
    MLIRFuncToEmitC
    MLIRFuncTransforms
    MLIRMathDialect
    MLIRReconcileUnrealizedCasts
    MLIRSCFToEmitC
    MLIRTransforms
    MLIRMathToEmitC
  )
