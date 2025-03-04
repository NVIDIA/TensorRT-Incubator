add_mlir_tensorrt_library(MLIRTensorRTCompilerStableHloToExecutable
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
    MLIRTensorRTCompilerClient
    MLIRTensorRTCUDAToExecutor
    MLIRTensorRTDropNestedModules
    MLIRTensorRTExecutorTransforms
    MLIRTensorRTHostToEmitC
    MLIRTensorRTHostToLLVM
    MLIRTensorRTMemRefToCUDA
    MLIRTensorRTOptionUtils
    MLIRTensorRTPipelines
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
    )
