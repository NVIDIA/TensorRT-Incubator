add_mlir_public_c_api_library(MLIRTensorRTCAPIExecutorTranslations
  ExecutorTranslations.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
  MLIRTensorRTTargetLua
  MLIRTensorRTCAPICommon
  )
