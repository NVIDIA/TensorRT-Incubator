add_mlir_tensorrt_library(MLIRTensorRTStableHloExtTransforms
    CanonicalizeDotGeneral.cpp
    ConstantFolding.cpp
    GatherToSlice.cpp
    CanonicalizeGather.cpp
    CanonicalizeScatter.cpp
    CanonicalizeShapes.cpp
    ExpandTuples.cpp

    LINK_LIBS PUBLIC
    MLIRRewrite
    MLIRTensorDialect
    MLIRTensorRTCommonUtils
    MLIRTensorRTStableHloExtUtils
    MLIRTensorRTConstantFoldingUtils
    StablehloOps

    DEPENDS
    MLIRTensorRTStableHloExtTransformsPassIncGen
)
