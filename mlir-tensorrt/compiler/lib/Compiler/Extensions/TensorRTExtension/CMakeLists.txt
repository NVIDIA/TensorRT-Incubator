add_mlir_tensorrt_library(
  MLIRTensorRTCompilerTensorRTExtension
  TensorRTExtension.cpp

  LINK_LIBS PUBLIC
    ${MTRTD_LIBS}
    MLIRTensorRTCompilerClient
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTStablehloInputPipeline
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTTensorRTToTensorRTRuntime
  )
