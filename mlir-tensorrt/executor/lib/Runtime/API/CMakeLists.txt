# The API library includes enum definitions, flatbuffer schema generated files,
# utilities and other functions that are critical for the interface/handshake
# between the Executor dialect and any runtime. It also contains the
# public-facing API headers. It shouldn't have any dependency on MLIR (except
# for perhaps LLVMSupport for right now) so that it can be used in the
# integrations. Anything specific to a runtime backend (e.g. Lua) also doens't
# belong here.
add_mlir_executor_runtime_library(MLIRTensorRTExecutorRuntimeAPI
  API.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTRuntimeSupport
  MLIRTensorRTSupportAllocators
  MLIRTensorRTSupportStatus
  flatbuffers
  sol2::sol2
  lua::core
  CUDA::cudart

  PRIVATE
  $<$<BOOL:${MLIR_EXECUTOR_ENABLE_NCCL}>:MPI::MPI_C>
  $<$<BOOL:${MLIR_EXECUTOR_ENABLE_NCCL}>:NCCL>


  DEPENDS
  MLIRTensorRTRuntimeExecutableFlatbufferGen
)
target_link_libraries(MLIRTensorRTExecutorRuntimeAPI PUBLIC DLPack::Headers)
