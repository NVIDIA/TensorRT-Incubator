add_mlir_tensorrt_library(MLIRTensorRTHostToEmitC
  HostToEmitC.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToEmitC
  MLIREmitCDialect
  MLIRSCFToEmitC
  MLIRTensorRTTensorRTToEmitC
  MLIRTensorRTExecutorDialect
  MLIRFuncToEmitC
  MLIRTensorRTCUDADialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTLLVMConversionUtils
  )
