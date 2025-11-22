add_mtrtd_library(MLIRTensorRTTensorRTDialectBase
  TensorRTBase.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRQuantDialect
  )

set(optional_deps)
if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND optional_deps MLIRTensorRTNvInferUtils)
endif()

add_mtrtd_library(MLIRTensorRTTensorRTDialect
  EinsumHelper.cpp
  TensorRT.cpp
  Verification.cpp
  TensorRTVersionCompatibility.cpp
  TensorKindOpInterfaceImpl.cpp
  TypeInferenceInterfaceImpls.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_LIST_DIR}

  DEPENDS
  MLIRTensorRTOpsIncGen
  MLIRTensorRTDialectAttributesIncGen
  MLIRTensorRTDialectEnumsIncGen
  MLIRTensorRTDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
    MLIRTensorRTCommonTensorKindInterface
    MLIRTensorRTConstantFoldingUtils
    MLIRTensorRTDialectIncludes
    MLIRTensorRTShapeUtils
    MLIRTensorRTTensorRTDialectBase
    ${optional_deps}
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowInterfaces
    MLIRDestinationStyleOpInterface
    MLIRDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRQuantDialect
    MLIRSupport
    MLIRTensorDialect
  )
