add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
    StableHloInputPipelines.cpp
    StablehloToExecutable.cpp
    TensorRTExtension.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTStablehloToExecutableIncGen

    LINK_LIBS PUBLIC
    MLIREmitCTransforms
    MLIRIR
    MLIRLLVMIRTransforms
    MLIRTensorRTChloToStablehloExt
    MLIRTensorRTCommonConvertToLoops
    MLIRTensorRTCompilerClient
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
    MLIRTensorRTStablehloToPlan
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTTargetLua
    MLIRTensorRTTargetTensorRT
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTTensorRTToTensorRTRuntime
    MLIRTensorRTTransforms
    StablehloLinalgTransforms
    )
