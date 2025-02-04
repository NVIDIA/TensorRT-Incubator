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
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRTensorRTCompilerClient
    MLIRTensorRTCUDAToExecutor
    MLIRTensorRTDropNestedModules
    MLIRTensorRTExecutorTransforms
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
