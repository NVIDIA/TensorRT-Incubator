add_mlir_public_c_api_library(MLIRTensorRTCAPISupportStatus
  Status.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTSupportStatus
  MLIRTensorRTCommonIncludes

  DEPENDS
  MLIRTensorRTStatusEnumIncGen
  )

if(TARGET obj.MLIRTensorRTCAPISupportStatus)
  target_link_libraries(obj.MLIRTensorRTCAPISupportStatus
    PUBLIC
    MLIRTensorRTCommonIncludes
  )
endif()