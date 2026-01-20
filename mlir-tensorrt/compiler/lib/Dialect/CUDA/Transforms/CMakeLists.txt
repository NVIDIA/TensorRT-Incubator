add_mlir_tensorrt_library(MLIRTensorRTCUDATransforms
  BufferizableOpInterfaceImpl.cpp
  CUDAExpandOps.cpp
  CUDAScheduleAsync.cpp
  CUDASimplifyStreamWait.cpp
  CUDAInsertHostSync.cpp

  DEPENDS
    MLIRTensorRTCUDATransformsPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerAnalysis
    MLIRTensorRTCUDADialect
    MLIRTensorRTStreamSchedulableOpInterface
    MLIRTensorRTTensorRTRuntimeDialect
  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRPass
    MLIRSupport
)
