# Note: currently we can't link `libKernelCompiler` since the test libs will
# link MLIR requirements directly.
mtrt_get_project_targets(Kernel LIBRARY_TYPE TEST_LIBS OUT_VAR MLIR_KERNEL_TEST_LIBS)
mtrt_get_project_targets(Kernel LIBRARY_TYPE LIBS OUT_VAR MLIR_KERNEL_LIBS)
set(extra_deps)

mtrt_add_tool(kernel-opt kernel-opt.cpp
  IGNORE_LINK_MTRT
  LINK_LIBS PRIVATE
  ${MLIR_KERNEL_TEST_LIBS}
  ${extra_deps}
  ${MLIR_KERNEL_LIBS}

  MLIR_LIBS PRIVATE
  MLIRFuncInlinerExtension
  MLIROptLib
  MLIRTransformDialectTransforms
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
)
