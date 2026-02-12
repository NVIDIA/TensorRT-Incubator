add_mlir_tensorrt_library(MLIRTensorRTKernelToCUDA
  KernelToCUDA.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDADialect
    MLIRTensorRTCUDAUtils
    MLIRTensorRTExecutorDialect
    MLIRTensorRTKernelDialect
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
