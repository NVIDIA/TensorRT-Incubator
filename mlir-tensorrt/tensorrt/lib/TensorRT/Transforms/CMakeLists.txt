include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_pdll_library(MLIRTensorRTTransformsActivationsPdllGen
 RaiseActivations.pdll
 RaiseActivations.pdll.h.inc
)

add_mlir_pdll_library(MLIRTensorRTTransformsNormalizationsPdllGen
 RaiseNormalizations.pdll
 RaiseNormalizations.pdll.h.inc
)

add_mtrtd_library(MLIRTensorRTTransforms
  ApplyWorkarounds.cpp
  BroadcastElimination.cpp
  ExpandOps.cpp
  InferPluginShapes.cpp
  LegalizeInt8.cpp
  Passes.cpp
  RaiseActivations.cpp
  RaiseNormalizations.cpp
  ReshapeElimination.cpp
  TransposeElimination.cpp

  DEPENDS
  MLIRTensorRTTransformsActivationsPdllGen
  MLIRTensorRTTransformsNormalizationsPdllGen
  MLIRTensorRTTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPass
  MLIRTensorRTAnalysis
  MLIRTensorRTCommonIncludes
  MLIRTensorRTConstantFoldingUtils
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  )
