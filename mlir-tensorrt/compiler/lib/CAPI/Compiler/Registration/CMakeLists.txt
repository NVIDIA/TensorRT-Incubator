mtrt_get_project_targets(MLIRTensorRT
  OUT_VAR mtrt_libs
  LIBRARY_TYPE LIBS)
mtrt_get_project_targets(MLIRTensorRT
  OUT_VAR mlir_libs
  LIBRARY_TYPE MLIR_LIBS)

mtrt_add_capi_library(MLIRTensorRTCAPIRegisterAllDialects
  RegisterAllDialects.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
    ${mtrt_libs}
    MLIRTensorRTCompilerStableHloToExecutable
    MLIRTensorRTCompilerTensorRTToExecutable
    MLIRTensorRTCUDAToLLVM
    MLIRTensorRTCUDATransforms
    MLIRTensorRTHostToEmitC
    MLIRTensorRTHostToLLVM
    MLIRTensorRTPlanToLLVM
    MLIRTensorRTTensorRTRuntimeTransforms
    MLIRTensorRTTensorRTToEmitC
  MLIR_LIBS PUBLIC
    ${mlir_libs}
    MLIRCAPIIR
    MLIREmitCTransforms
    MLIRFuncInlinerExtension
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTransformDialect
    )
