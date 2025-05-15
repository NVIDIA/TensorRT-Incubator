add_subdirectory(TensorRTRuntime)

if(MLIR_TRT_ENABLE_EXECUTOR)
  add_subdirectory(CUDA)
endif()

if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(Plan)
  add_subdirectory(StablehloExt)
endif()
