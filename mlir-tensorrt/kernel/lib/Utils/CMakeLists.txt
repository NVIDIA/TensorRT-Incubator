add_compile_definitions(
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}"
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}"
  "MLIR_TRT_COMPILE_TIME_CUDA_VERSION_PATCH=${CUDAToolkit_VERSION_PATCH}"
)

add_mlir_kernel_library(MLIRKernelTilingUtils
  TilingUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRLinalgDialect
)

add_mlir_kernel_library(MLIRKernelLinalgUtils
  StructuredOpsUtils.cpp
  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRTransformUtils
)

add_mlir_kernel_library(MLIRKernelTensorUtils
  TensorUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRKernelLinalgUtils
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRDialectUtils
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTilingInterface
  )

add_mlir_kernel_library(MLIRKernelOutliningUtils
  OutliningUtils.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
    MLIRTensorRTRegionUtils
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRBufferizationDialect
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransformUtils
  )


add_mlir_kernel_library(MLIRKernelScatterUtils
  ScatterUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRQuantDialect
  )

add_mlir_kernel_library(MLIRKernelCUDAUtils
  CUDAUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRNVGPUDialect
    MLIRNVVMDialect
  )
