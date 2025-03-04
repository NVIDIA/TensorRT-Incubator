add_mtrtd_library(MLIRTensorRTTensorRTDialectBase
  TensorRTBase.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuantDialect
  )


add_mtrtd_library(MLIRTensorRTDialect
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
  MLIRArithDialect
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRDestinationStyleOpInterface
  MLIRTensorDialect
  MLIRTensorRTInterfaces
  MLIRTensorRTNvInferUtils
  MLIRTensorRTShapeUtils
  MLIRTensorRTConstantFoldingUtils
  MLIRTensorRTTensorRTDialectBase
  )
