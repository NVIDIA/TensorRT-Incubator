set(LLVM_LINK_COMPONENTS
  Support
  )
# Note: tablegen always linked statically, not against libMLIR.so.
mtrt_add_tool(tensorrt-tblgen tensorrt-tblgen.cpp
  IGNORE_LINK_MTRT
  MLIR_LIBS PRIVATE
  MLIRTblgenLib
)
