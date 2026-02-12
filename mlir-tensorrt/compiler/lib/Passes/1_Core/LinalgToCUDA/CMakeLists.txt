add_mlir_tensorrt_library(MLIRTensorRTLinalgToCUDA
  LinalgToCUDA.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDADialect
    MLIRTensorRTCUDAUtils
    MLIRTensorRTPlanDialect

  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRTransformUtils
)
