add_subdirectory(Registration)

mtrt_add_capi_library(MLIRTensorRTCAPICompiler
  Compiler.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerClient
    MLIRTensorRTCAPISupportStatus
    MLIRTensorRTCAPICommon
  MLIR_LIBS PUBLIC
    MLIRCAPIIR
)
