add_llvm_executable(executor-tblgen executor-tblgen.cpp)

llvm_update_compile_flags(executor-tblgen)
target_link_libraries(executor-tblgen PRIVATE
  MLIRTblgenLib
  )

mlir_check_all_link_libraries(executor-tblgen)
