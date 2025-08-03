add_mlir_tool(mlir-tensorrt-opt
  mlir-tensorrt-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorRTConversionPassIncGen
  MLIROptLib
  ${MLIR_TENSORRT_LIBS}

  SUPPORT_PLUGINS
)

target_link_libraries(mlir-tensorrt-opt PRIVATE
  MLIROptLib
  ${MLIR_TENSORRT_LIBS}
  )

llvm_update_compile_flags(mlir-tensorrt-opt)
mlir_check_link_libraries(mlir-tensorrt-opt)
export_executable_symbols_for_plugins(mlir-tensorrt-opt)
