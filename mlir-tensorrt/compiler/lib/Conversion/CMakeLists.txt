add_subdirectory(TensorRTCommon)

if(MLIR_TRT_ENABLE_HLO)
  add_subdirectory(StablehloToPlan)
  add_subdirectory(StablehloToTensorRT)
  add_subdirectory(StablehloScalarToArith)
  add_subdirectory(StablehloToScf)
  add_subdirectory(ChloToStablehloExt)
endif()

add_subdirectory(CUDAToExecutor)
add_subdirectory(CUDAToLLVM)
add_subdirectory(HostToEmitC)
add_subdirectory(HostToLLVM)
add_subdirectory(LLVMCommon)
add_subdirectory(LowerLinalgCopies)
add_subdirectory(MemRefToCUDA)
add_subdirectory(PlanToExecutor)
add_subdirectory(PlanToLLVM)
add_subdirectory(TensorRTRuntimeToExecutor)
add_subdirectory(TensorRTRuntimeToLLVM)
add_subdirectory(TensorRTToEmitC)
add_subdirectory(TensorRTToTensorRTRuntime)
