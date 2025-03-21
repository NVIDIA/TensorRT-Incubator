add_mlir_tensorrt_library(MLIRTensorRTCompilerTensorRTToExecutable
    TensorRTToExecutable.cpp
    Passes.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTTensorRTToExecutableIncGen

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
    MLIRTensorRTPlanToExecutor
    MLIRTensorRTPlanTransforms
    MLIRTensorRTRuntimeToExecutor
    MLIRTensorRTSCFDetensorizeLoops
    MLIRTensorRTTargetLua
    MLIRTensorRTTargetTensorRT
    MLIRTensorRTTensorRTToTensorRTRuntime
    MLIRTensorRTTransforms
    )
