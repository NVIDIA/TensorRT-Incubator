include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_pdll_library(MLIRTensorRTTransformsActivationsPdllGen
 RaiseActivations.pdll
 RaiseActivations.pdll.h.inc
)

add_mlir_pdll_library(MLIRTensorRTTransformsNormalizationsPdllGen
 RaiseNormalizations.pdll
 RaiseNormalizations.pdll.h.inc
)

add_mlir_library(MLIRTensorRTTransforms
  ApplyBugWorkarounds.cpp
  BroadcastElimination.cpp
  ExpandOps.cpp
  LegalizeInt8.cpp
  TransposeElimination.cpp
  ReshapeElimination.cpp
  RaiseActivations.cpp
  RaiseNormalizations.cpp
  Passes.cpp
  InferPluginShapes.cpp

  DEPENDS
  MLIRTensorRTTransformsPassIncGen
  MLIRTensorRTTransformsActivationsPdllGen
  MLIRTensorRTTransformsNormalizationsPdllGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTensorRTAnalysis
  MLIRTensorRTConstantFoldingUtils
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTUtils
  MLIRTransforms
  )
