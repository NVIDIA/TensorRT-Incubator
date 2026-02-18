if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(TensorRTDynamicLoader)
endif()

add_compile_definitions(
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}"
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}"
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_PATCH=${CUDAToolkit_VERSION_PATCH}"
)

add_mlir_tensorrt_library(MLIRTensorRTCUDAToolkitUtils
  CUDAUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRNVGPUDialect
    MLIRNVVMDialect
  )

add_mlir_tensorrt_library(MLIRTensorRTOwningModuleRef
  OwningModuleRef.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
  )

add_mlir_tensorrt_library(MLIRTensorRTRegionUtils
  RegionUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRTransformUtils
  )

add_mlir_tensorrt_library(MLIRTensorRTUtilsShapeInfo
  ShapeInfo.cpp
  ModuleUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRDialectUtils
    MLIRFuncDialect
  )

add_mlir_tensorrt_library(MLIRTensorRTModuleLikePass
  ModuleLikePass.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRPass
  )
