set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_tablegen(mtrt-tblgen MTRT
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  EXPORT MTRT
  mtrt-tblgen.cpp
  )

llvm_update_compile_flags(mtrt-tblgen)
target_link_libraries(mtrt-tblgen PRIVATE
  MLIRTblgenLib
  )

mlir_check_all_link_libraries(mtrt-tblgen)

