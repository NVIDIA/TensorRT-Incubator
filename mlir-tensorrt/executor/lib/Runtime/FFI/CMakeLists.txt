add_mlir_executor_runtime_library(MLIRTensorRTRuntimeFFIAPI
  FFI.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTSupportStatus
    DLPack::Headers
    tvm_ffi_shared
  )
# The FFI API provider requires exceptions since foreign functions may throw.
# The handler dispatch just ensures that these exceptions don't leak out of this
# library context.
target_compile_options(
  obj.MLIRTensorRTRuntimeFFIAPI PRIVATE
  -fexceptions
  -Wno-non-virtual-dtor
)
