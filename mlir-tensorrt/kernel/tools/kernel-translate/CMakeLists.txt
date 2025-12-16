set(LLVM_LINK_COMPONENTS
  Support
  )

mtrt_add_tool(kernel-translate kernel-translate.cpp
  IGNORE_LINK_MTRT
  MLIR_LIBS PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIRTargetCpp
  )
