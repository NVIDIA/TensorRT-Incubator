set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(executor-translate
  executor-translate.cpp
  )
llvm_update_compile_flags(executor-translate)
target_link_libraries(executor-translate
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIRTensorRTTargetLua
  )

mlir_check_link_libraries(executor-translate)
