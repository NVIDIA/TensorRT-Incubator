mtrt_add_tool(mlir-tensorrt-translate mlir-tensorrt-translate.cpp
  LINK_LIBS PRIVATE
  ${MLIR_TENSORRT_LIBS}
  MLIR_LIBS PRIVATE
  MLIRTranslateLib
  MLIRToLLVMIRTranslationRegistration
  )
