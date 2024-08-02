add_mlir_library(MLIRTensorRTNvInferUtils
  NvInferPluginUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  CUDA::cudart
  TensorRTHeaderOnly
  MLIRTRTTensorRTDynamicLoader
)

add_mlir_library(MLIRTensorRTConstantFoldingUtils
  PARTIAL_SOURCES_INTENDED
  ConstantFoldUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
)

add_mlir_library(MLIRTensorRTShapeUtils
  ShapeUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  )

add_mlir_library(MLIRTensorRTOptionUtils
  Options.cpp
  PARTIAL_SOURCES_INTENDED
  )