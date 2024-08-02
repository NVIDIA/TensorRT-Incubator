add_subdirectory(TensorRTCommon)

if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(StablehloToTensorRT)
  add_subdirectory(StablehloScalarToArith)
  add_subdirectory(StablehloToScf)
endif()

if(MLIR_TRT_TARGET_CPP)
  add_subdirectory(TensorRTToEmitC)
endif()

if(MLIR_TRT_ENABLE_EXECUTOR)
  add_subdirectory(MemRefToCUDA)
  add_subdirectory(TensorRTToTensorRTRuntime)
  add_subdirectory(CUDAToExecutor)
  add_subdirectory(PlanToExecutor)
  add_subdirectory(TensorRTRuntimeToExecutor)
endif()
