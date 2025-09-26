add_subdirectory(Registration)

mtrt_add_capi_library(MLIRTensorRTCAPICompiler
  Compiler.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_TENSORRT_ROOT_DIR}/include

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTensorRTCompilerStableHloToExecutable
  MLIRTensorRTCAPISupportStatus
  MLIRTensorRTCAPICommon
)
