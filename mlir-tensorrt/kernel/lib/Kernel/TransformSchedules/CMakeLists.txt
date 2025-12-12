add_mlir_kernel_library(MLIRTensorRTKernelTransformSchedules
  FallbackSchedule.cpp
  InitialTransformSchedule.cpp
  ScatterSchedule.cpp
  TransformBuilder.cpp
  TransformSchedules.cpp

  DEPENDS
  MLIRTensorRTKernelTransformSchedulesPassIncGen

  LINK_LIBS PUBLIC
    MLIRKernelTilingUtils
    MLIRTensorRTCommonToLoopsOpInterface
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelTransformOps
    MLIRTensorRTSupportDeviceInfo
    MLIRTensorRTSupportStatus
  MLIR_LIBS PUBLIC
    MLIRBufferizationTransformOps
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRTensorTransformOps
    MLIRTransformLoopExtension
    MLIRVectorTransformOps
)
target_include_directories(obj.MLIRTensorRTKernelTransformSchedules
  PRIVATE
  ${MLIR_TENSORRT_ROOT_DIR}/executor/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)
