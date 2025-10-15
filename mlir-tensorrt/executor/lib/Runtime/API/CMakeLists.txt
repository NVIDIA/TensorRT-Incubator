add_mlir_executor_runtime_library(MLIRTensorRTRuntimeExecutableInterface
  Executable.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorRTRuntimeExecutableFlatbufferGen

  LINK_LIBS PUBLIC
  flatbuffers
  MLIRExecutorIncludes
  MLIRTensorRTSupportStatus
  )

# The API library includes enum definitions, flatbuffer schema generated files,
# utilities and other functions that are critical for the interface/handshake
# between the Executor dialect and any runtime. It also contains the
# public-facing API headers. It shouldn't have any dependency on MLIR (except
# for perhaps LLVMSupport for right now) so that it can be used in the
# integrations. Anything specific to a runtime backend (e.g. Lua) also doens't
# belong here.
add_mlir_executor_runtime_library(MLIRTensorRTExecutorRuntimeAPI
  API.cpp
  MemRefABI.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorRTRuntimeExecutableFlatbufferGen

  LINK_LIBS PUBLIC
  lua::core
  MLIRTensorRTRuntimeExecutableInterface
  MLIRTensorRTRuntimeSupport
  MLIRTensorRTSupportAllocators
  MLIRTensorRTSupportStatus
  MLIRTensorRTSupportStridedCopy
  ${MLIR_TRT_MPI_TARGET}
  ${MLIR_TRT_NCCL_TARGET}
  ${MLIR_TRT_CUDA_TARGET}
  )
