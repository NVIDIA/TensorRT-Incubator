if(MLIR_TRT_TARGET_TENSORRT)
  add_mtrtd_library(MLIRTensorRTNvInferUtils
    NvInferPluginUtils.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    CUDA::cudart
    MLIRArithDialect
    MLIRIR
    MLIRTensorRTCommonIncludes
    MLIRTensorRTTensorRTDialectBase
    MLIRTRTTensorRTDynamicLoader
    TensorRTHeaderOnly
  )
endif()

add_mtrtd_library(MLIRTensorRTConstantFoldingUtils
  PARTIAL_SOURCES_INTENDED
  ConstantFoldUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTensorRTDialectIncludes
)

add_mtrtd_library(MLIRTensorRTShapeUtils
  ShapeUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  MLIRTensorRTDialectIncludes
  )
