mtrt_add_tool(mlir-tensorrt-compiler mlir-tensorrt-compiler.cpp
  LINK_LIBS PRIVATE ${MLIR_TENSORRT_LIBS}
)

