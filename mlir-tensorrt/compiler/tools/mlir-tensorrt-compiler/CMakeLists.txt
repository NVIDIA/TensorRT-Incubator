mtrt_get_project_targets(MLIRTensorRT OUT_VAR MLIR_TENSORRT_LIBS)

add_mlir_tool(mlir-tensorrt-compiler mlir-tensorrt-compiler.cpp)

llvm_update_compile_flags(mlir-tensorrt-compiler)
target_link_libraries(mlir-tensorrt-compiler PRIVATE
  ${MLIR_TENSORRT_LIBS}
  )

mlir_check_all_link_libraries(mlir-tensorrt-compiler)

install(TARGETS mlir-tensorrt-compiler
    RUNTIME
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT mlir-tensorrt-compiler
  )