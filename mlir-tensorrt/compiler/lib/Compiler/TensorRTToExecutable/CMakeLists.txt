mtrt_get_project_targets(
    TensorRTDialect
    OUT_VAR MTRTD_LIBS
)

add_mlir_tensorrt_library(MLIRTensorRTCompilerTensorRTToExecutable
    TensorRTToExecutable.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTTensorRTToExecutableIncGen

    LINK_LIBS PUBLIC
      MLIRTensorRTCompilerClient
      MLIRTensorRTCUDAToExecutor
      MLIRTensorRTDropNestedModules
      MLIRTensorRTExecutorTransforms
      MLIRTensorRTHostToLLVM
      MLIRTensorRTMemRefToCUDA
      MLIRTensorRTPlanToExecutor
      MLIRTensorRTPlanTransforms
      MLIRTensorRTRegionUtils
      MLIRTensorRTRuntimeToExecutor
      MLIRTensorRTSCFDetensorizeLoops
      MLIRTensorRTTargetLua
      MLIRTensorRTTensorRTBackend
      MLIRTensorRTTensorRTToTensorRTRuntime
      MLIRTensorRTTransforms
      ${MTRTD_LIBS}
    MLIR_LIBS PUBLIC
      MLIREmitCTransforms
      MLIRIR
      MLIRLLVMDialect
      MLIRLLVMIRTransforms
    )
