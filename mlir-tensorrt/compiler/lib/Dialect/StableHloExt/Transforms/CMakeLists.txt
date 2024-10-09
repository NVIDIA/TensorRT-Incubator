add_mlir_tensorrt_library(MLIRTensorRTStableHloExtTransforms
    CanonicalizeDotGeneral.cpp
    ConstantFolding.cpp
    GatherToSlice.cpp
    CanonicalizeGather.cpp
    CanonicalizeScatter.cpp
    CanonicalizeShapes.cpp
    ExpandTuples.cpp
    LowerSpecialCustomCalls.cpp

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
