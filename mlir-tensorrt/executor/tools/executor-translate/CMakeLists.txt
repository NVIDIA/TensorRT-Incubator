set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_executor_tool(executor-translate executor-translate.cpp
 MLIR_LIBS PRIVATE
 MLIRIR
 MLIRParser
 MLIRPass
 MLIRTranslateLib
 MLIRSupport

 LINK_LIBS PRIVATE
 MLIRTensorRTTargetLua
 )
