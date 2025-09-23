add_mlir_tool(mlir-tensorrt-lsp-server
  mlir-tensorrt-lsp-server.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorRTInitAllDialects
  MLIRLspServerLib
)

target_link_libraries(mlir-tensorrt-lsp-server PRIVATE
  MLIRTensorRTInitAllDialects
  MLIRLspServerLib
)
llvm_update_compile_flags(mlir-tensorrt-lsp-server)
mlir_check_link_libraries(mlir-tensorrt-lsp-server)

set(should_exclude_from_all TRUE)
if(PROJECT_IS_TOP_LEVEL)
  set(should_exclude_from_all FALSE)
endif()
set_target_properties(mlir-tensorrt-lsp-server PROPERTIES EXCLUDE_FROM_ALL ${should_exclude_from_all})
