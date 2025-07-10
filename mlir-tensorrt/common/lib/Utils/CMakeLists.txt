if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(TensorRTDynamicLoader)
endif()


add_mlir_tensorrt_library(MLIRTensorRTOwningModuleRef
  OwningModuleRef.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  )
