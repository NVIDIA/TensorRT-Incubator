add_mlir_tool(mlir-tensorrt-lsp-server
  mlir-tensorrt-lsp-server.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${MLIR_TENSORRT_LIBS}
  MLIRLspServerLib
)

target_link_libraries(mlir-tensorrt-lsp-server PRIVATE
  ${MLIR_TENSORRT_LIBS}
  MLIRLspServerLib
)
llvm_update_compile_flags(mlir-tensorrt-lsp-server)
mlir_check_link_libraries(mlir-tensorrt-lsp-server)
set_target_properties(mlir-tensorrt-lsp-server PROPERTIES EXCLUDE_FROM_ALL TRUE)