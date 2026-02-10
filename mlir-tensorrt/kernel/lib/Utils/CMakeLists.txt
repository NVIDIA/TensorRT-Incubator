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
