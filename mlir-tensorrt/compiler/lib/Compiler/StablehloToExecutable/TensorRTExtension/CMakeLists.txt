add_mlir_tensorrt_library(
  MLIRTensorRTStableHloToExecutableTensorRTExtension
  TensorRTExtension.cpp

  LINK_LIBS PUBLIC
    ${MTRTD_LIBS}
    MLIRTensorRTCompilerClient
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTCompilerStableHloToExecutable
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTTensorRTToTensorRTRuntime
  )
