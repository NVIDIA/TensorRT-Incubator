set(LLVM_LINK_COMPONENTS
  Support
  )
# Note: tablegen always linked statically, not against libMLIR.so.
add_tablegen(tensorrt-tblgen MTRT_TRT_DIALECT
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  EXPORT MTRTTargets
  tensorrt-tblgen.cpp
)
llvm_update_compile_flags(tensorrt-tblgen)
target_link_libraries(tensorrt-tblgen PRIVATE
  MLIRTblgenLib
)
mlir_check_all_link_libraries(tensorrt-tblgen)
