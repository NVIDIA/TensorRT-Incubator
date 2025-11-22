add_mlir_tensorrt_library(MLIRTensorRTStableHloExtTransforms
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
