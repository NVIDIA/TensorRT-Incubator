if(MLIR_TRT_TARGET_TENSORRT)
  add_mtrtd_library(MLIRTensorRTNvInferUtils
    NvInferPluginUtils.cpp
    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
      ${MLIR_TRT_CUDA_TARGET}
      MLIRTensorRTCommonIncludes
      MLIRTensorRTTensorRTDialectBase
      TensorRTHeaderOnly
    MLIR_LIBS PUBLIC
      MLIRArithDialect
      MLIRIR
      MLIRTRTTensorRTDynamicLoader
  )
endif()

add_mtrtd_library(MLIRTensorRTConstantFoldingUtils
  PARTIAL_SOURCES_INTENDED
  ConstantFoldUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTDialectIncludes
  MLIR_LIBS PUBLIC
    MLIRDialectUtils
    MLIRIR
)

add_mtrtd_library(MLIRTensorRTShapeUtils
  ShapeUtils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTDialectIncludes
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRDialectUtils
  )
