mtrt_get_project_targets(
    TensorRTDialect
    OUT_VAR MTRTD_LIBS
)

add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StableHloInputPipelines.cpp
    StablehloToExecutable.cpp
    TensorRTExtension.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTStablehloToExecutableIncGen

    LINK_LIBS PUBLIC
    ${MTRTD_LIBS}
    MLIREmitCTransforms
    MLIRIR
    MLIRLLVMIRTransforms
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
    MLIRTensorRTMemRefToCUDA
    MLIRTensorRTPlanToExecutor
    MLIRTensorRTPlanTransforms
    MLIRTensorRTRuntimeToExecutor
    MLIRTensorRTSCFDetensorizeLoops
    MLIRTensorRTStablehloToLinalg
    MLIRTensorRTStablehloToPlan
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTTargetLua
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTTensorRTToTensorRTRuntime
    MLIRTensorRTTransformsUnrollForLoops
    StablehloLinalgTransforms
    )
