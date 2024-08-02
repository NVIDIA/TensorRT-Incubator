add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Utils)

if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(Target)
endif()
