add_mlir_tensorrt_library(MLIRTensorRTStableHloExtTransforms
    CanonicalizeDotGeneral.cpp
    CanonicalizeGather.cpp
    CanonicalizeScatter.cpp
    CanonicalizeShapes.cpp
    ConstantFolding.cpp
    ExpandTuples.cpp
    GatherToSlice.cpp
    LowerSpecialCustomCalls.cpp
    StablehloInputPreprocessing.cpp
    StablehloPrepareConvolution.cpp
    StablehloPrepareScatter.cpp
    StablehloRaiseQDQ.cpp

    LINK_LIBS PUBLIC
    ChloOps
    MLIRRewrite
    MLIRTensorDialect
    MLIRExecutorCommonUtils
    MLIRTensorRTConstantFoldingUtils
    MLIRTensorRTShapeUtils
    MLIRTensorRTStableHloExtUtils
    StablehloOps
    StablehloPasses

    DEPENDS
    MLIRTensorRTStableHloExtTransformsPassIncGen
)
