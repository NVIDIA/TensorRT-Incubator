macro(mtrt_llvm_project)
  CPMAddPackage(
    ${ARGN}
    )
  set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
  set(LLVM_CMAKE_DIR "${llvm_project_BINARY_DIR}/lib/cmake/llvm")
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(LLVMConfig)
  include(MLIRConfig)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
endmacro()
