add_mlir_dialect(TensorRTRuntimeOps trtrt)
add_mlir_tensorrt_doc(TensorRTRuntimeDialect
  SRC TensorRTRuntimeOps.td
  OUTPUT_FILE docs/Dialects/TensorRTRuntimeDialect.md
  COMMAND -gen-dialect-doc -dialect=trtrt
  )
