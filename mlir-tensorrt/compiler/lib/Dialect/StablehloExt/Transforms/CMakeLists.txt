add_mlir_tensorrt_library(MLIRTensorRTStableHloExtTransforms
  CastPatterns.cpp
  CanonicalizeConvolution.cpp
  CanonicalizeDotGeneral.cpp
  CanonicalizeGather.cpp
  CanonicalizeScatter.cpp
  CanonicalizeShapes.cpp
  ConstantFolding.cpp
  ExpandTuples.cpp
  GatherToSlice.cpp
  LowerSpecialCustomCalls.cpp
  MaterializeDenseResourceElementsAttr.cpp
  Simplifications.cpp
  StablehloRaiseQDQ.cpp
  TargetSpecificOptimizations.cpp

  LINK_LIBS PUBLIC
    ChloOps
    MLIRExecutorCommonUtils
    MLIRTensorRTConstantFoldingUtils
    MLIRTensorRTShapeUtils
    MLIRTensorRTStableHloExtUtils
    StablehloOps
    StablehloPasses
    StablehloOptimizationPasses
  MLIR_LIBS PUBLIC
    MLIRRewrite
    MLIRTensorDialect
  DEPENDS
    MLIRTensorRTStableHloExtTransformsPassIncGen
)
