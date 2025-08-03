mtrt_get_project_targets(MLIRTensorRT OUT_VAR MLIR_TENSORRT_LIBS)

add_mlir_tensorrt_public_c_api_library(MLIRTensorRTCAPIRegisterAllDialects
  RegisterAllDialects.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIREmitCTransforms
    MLIRFuncInlinerExtension
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorRTCompilerStableHloToExecutable
    MLIRTensorRTCompilerTensorRTToExecutable
    MLIRTensorRTCUDAToLLVM
    MLIRTensorRTCUDATransforms
    MLIRTensorRTHostToEmitC
    MLIRTensorRTHostToLLVM
    MLIRTensorRTPlanToLLVM
    MLIRTensorRTTensorRTRuntimeTransforms
    MLIRTensorRTTensorRTToEmitC
    MLIRTransformDialect
    ${MLIR_TENSORRT_LIBS}
    )
