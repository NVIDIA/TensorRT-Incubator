if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(StablehloExt)
  add_subdirectory(Plan)
endif()

add_subdirectory(TensorRTRuntime)

if(MLIR_TRT_ENABLE_EXECUTOR)
  add_subdirectory(CUDA)
endif()
