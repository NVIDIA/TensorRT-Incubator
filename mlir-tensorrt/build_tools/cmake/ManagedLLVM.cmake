macro(mtrt_llvm_project)
  CPMAddPackage(
    NAME llvm_project
    URL "https://github.com/llvm/llvm-project/archive/${MLIR_TRT_LLVM_COMMIT}.zip"
    EXCLUDE_FROM_ALL TRUE
    SOURCE_SUBDIR "llvm"
    PATCHES "${CMAKE_SOURCE_DIR}/build_tools/llvm-project.patch"
    OPTIONS
      "LLVM_ENABLE_PROJECTS mlir"
      "MLIR_ENABLE_BINDINGS_PYTHON ${MLIR_TRT_ENABLE_PYTHON}"
      "LLVM_TARGETS_TO_BUILD host"
      "LLVM_ENABLE_ASSERTIONS ${MLIR_TRT_ENABLE_ASSERTIONS}"
      "LLVM_USE_LINKER ${MLIR_TRT_USE_LINKER}"
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
