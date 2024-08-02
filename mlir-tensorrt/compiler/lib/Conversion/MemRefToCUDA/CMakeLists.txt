add_mlir_tensorrt_library(MLIRTensorRTMemRefToCUDA MemRefToCUDA.cpp
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRTensorRTCUDADialect
    MLIRTensorRTPlanDialect

    DEPENDS
    MLIRTensorRTConversionPassIncGen
)
