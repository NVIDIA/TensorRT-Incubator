add_mlir_public_c_api_library(MLIRExecutorCAPIRuntimeCore
  CoreModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  LINK_LIBS PUBLIC
  MLIRTensorRTCAPISupportStatus
  )
