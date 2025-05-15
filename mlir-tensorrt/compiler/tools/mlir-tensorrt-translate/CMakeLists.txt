add_mlir_tool(mlir-tensorrt-translate
  mlir-tensorrt-translate.cpp
  PARTIAL_SOURCES_INTENDED
  )

target_link_libraries(mlir-tensorrt-translate PRIVATE
  MLIRTranslateLib
  MLIRTargetCpp
  ${MLIR_TENSORRT_LIBS}
  )
llvm_update_compile_flags(mlir-tensorrt-translate)
mlir_check_link_libraries(mlir-tensorrt-translate)