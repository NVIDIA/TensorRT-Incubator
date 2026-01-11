add_subdirectory(CUDA)

if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(Plan)
  add_subdirectory(StablehloExt)
endif()
