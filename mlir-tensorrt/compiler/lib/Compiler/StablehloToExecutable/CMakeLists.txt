mtrt_get_project_targets(
    TensorRTDialect
    OUT_VAR MTRTD_LIBS
)

add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StableHloInputPipeline.cpp
    StablehloToExecutable.cpp
    TensorRTExtension.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
      ${MTRTD_LIBS}
      MLIRTensorRTChloToStablehloExt
      MLIRTensorRTCommonConvertToLoops
      MLIRTensorRTCompilerClient
      MLIRTensorRTCompilerIncludes
      MLIRTensorRTCUDAToExecutor
      MLIRTensorRTDropNestedModules
      MLIRTensorRTExecutorTransforms
      MLIRTensorRTHostBackend
      MLIRTensorRTHostToEmitC
      MLIRTensorRTHostToLLVM
      MLIRTensorRTLinalgElementwiseFusion
      MLIRTensorRTMemRefToCUDA
      MLIRTensorRTPlanToExecutor
      MLIRTensorRTPlanTransforms
      MLIRTensorRTRuntimeToExecutor
      MLIRTensorRTSCFDetensorizeLoops
      MLIRTensorRTStablehloToArith
      MLIRTensorRTStablehloToLinalg
      MLIRTensorRTStablehloToPlan
      MLIRTensorRTStablehloToSCF
      MLIRTensorRTStablehloToTensorRT
      MLIRTensorRTTargetLua
      MLIRTensorRTTensorRTBackend
      MLIRTensorRTTensorRTToTensorRTRuntime
      MLIRTensorRTTransformsSCFFloatStrengthReduce
      MLIRTensorRTTransformsUnrollForLoops
      StablehloLinalgTransforms
    MLIR_LIBS PUBLIC
      MLIRArithTransforms
      MLIREmitCTransforms
      MLIRIR
      MLIRLLVMIRTransforms
      MLIRMathToEmitC
    )

add_subdirectory(KernelGenExtension)
