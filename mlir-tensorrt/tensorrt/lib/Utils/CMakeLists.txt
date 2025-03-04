add_mtrtd_library(MLIRTensorRTNvInferUtils
  NvInferPluginUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  CUDA::cudart
  TensorRTHeaderOnly
  MLIRTRTTensorRTDynamicLoader
  MLIRTensorRTTensorRTDialectBase
)

add_mtrtd_library(MLIRTensorRTConstantFoldingUtils
  PARTIAL_SOURCES_INTENDED
  ConstantFoldUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
)

add_mtrtd_library(MLIRTensorRTShapeUtils
  ShapeUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  )

add_mtrtd_library(MLIRTensorRTOptionUtils
  Options.cpp
  PARTIAL_SOURCES_INTENDED
  )