set(link_deps)
if(MLIR_TRT_LINK_MTRT_DYLIB)
  list(APPEND link_deps MTRT)
else()
  list(APPEND link_deps MLIRTensorRTInitAllDialects)
endif()

mtrt_add_tool(mlir-tensorrt-lsp-server mlir-tensorrt-lsp-server.cpp
  LINK_LIBS PRIVATE
  ${link_deps}
  MLIR_LIBS PRIVATE
  MLIRLspServerLib
)

set(should_exclude_from_all TRUE)
if(PROJECT_IS_TOP_LEVEL)
  set(should_exclude_from_all FALSE)
endif()
set_target_properties(mlir-tensorrt-lsp-server PROPERTIES EXCLUDE_FROM_ALL ${should_exclude_from_all})
