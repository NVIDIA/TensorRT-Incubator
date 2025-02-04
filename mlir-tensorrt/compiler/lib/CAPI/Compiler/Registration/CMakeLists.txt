add_mlir_tensorrt_public_c_api_library(MLIRTensorRTCAPIRegisterAllDialects
  RegisterAllDialects.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIREmitCTransforms
    MLIRFuncInlinerExtension
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorRTCompilerStableHloToExecutable
    MLIRTensorRTCUDAToLLVM
    MLIRTensorRTCUDATransforms
    MLIRTensorRTHostToLLVM
    MLIRTensorRTPlanToLLVM
    MLIRTensorRTTensorRTRuntimeTransforms
    MLIRTensorRTTensorRTToEmitC
    MLIRTransformDialect
    )
