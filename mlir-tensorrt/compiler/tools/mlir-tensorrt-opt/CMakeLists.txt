
set(link_deps ${MLIR_TENSORRT_LIBS})
mtrt_add_tool(mlir-tensorrt-opt
   mlir-tensorrt-opt.cpp
   PARTIAL_SOURCES_INTENDED

   DEPENDS
   ${link_deps}

   SUPPORT_PLUGINS

   LINK_LIBS PRIVATE
   ${link_deps}
   MLIR_LIBS PRIVATE
   MLIROptLib
)

export_executable_symbols_for_plugins(mlir-tensorrt-opt)
