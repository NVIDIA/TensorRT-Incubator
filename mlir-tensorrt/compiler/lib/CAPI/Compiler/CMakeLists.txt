add_subdirectory(Registration)

mtrt_add_capi_library(MLIRTensorRTCAPICompiler
  Compiler.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensorRTCompilerStableHloToExecutable
  MLIRTensorRTCAPISupportStatus
  MLIRTensorRTCAPICommon
)
