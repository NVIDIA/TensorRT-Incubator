
set(LLVM_LINK_COMPONENTS
  Support
  )
add_llvm_executable(tensorrt-tblgen tensorrt-tblgen.cpp)
llvm_update_compile_flags(tensorrt-tblgen)
target_link_libraries(tensorrt-tblgen PRIVATE
  MLIRTblgenLib
  )

mlir_check_all_link_libraries(tensorrt-tblgen)
