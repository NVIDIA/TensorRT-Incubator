mtrt_add_capi_library(MLIRTensorRTCAPISupportStatus
  Status.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
  MLIRTensorRTCommonIncludes

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
  )
