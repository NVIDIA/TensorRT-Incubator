add_subdirectory(CUDA)
add_subdirectory(Plan)
add_subdirectory(TensorRTRuntime)

if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(StablehloExt)
endif()

