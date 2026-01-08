add_mlir_tensorrt_library(MLIRTensorRTMemRefToCUDA MemRefToCUDA.cpp
    LINK_LIBS PUBLIC
      MLIRTensorRTCUDADialect
      MLIRTensorRTCUDAUtils
      MLIRTensorRTPlanDialect
    MLIR_LIBS PUBLIC
      MLIRIR
      MLIRMemRefDialect
      MLIRPass
    DEPENDS
    MLIRTensorRTConversionPassIncGen
)
