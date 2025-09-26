mtrt_add_capi_library(MLIRTensorRTCAPIExecutorTranslations
  ExecutorTranslations.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
  MLIRTensorRTTargetLua
  MLIRTensorRTCAPICommon
  )
