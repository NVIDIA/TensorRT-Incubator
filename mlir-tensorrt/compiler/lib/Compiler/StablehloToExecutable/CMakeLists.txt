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
    MLIRTensorRTCompilerClient
    MLIRTensorRTCUDAToExecutor
    MLIRTensorRTDropNestedModules
    MLIRTensorRTExecutorTransforms
    MLIRTensorRTHostToEmitC
    MLIRTensorRTHostToLLVM
    MLIRTensorRTMemRefToCUDA
    MLIRTensorRTPlanToExecutor
    MLIRTensorRTPlanTransforms
    MLIRTensorRTRuntimeToExecutor
    MLIRTensorRTSCFDetensorizeLoops
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTTargetLua
    MLIRTensorRTTargetTensorRT
    MLIRTensorRTTensorRTToTensorRTRuntime
    MLIRTensorRTTransforms
    StablehloLinalgTransforms
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTHostBackend
    )
