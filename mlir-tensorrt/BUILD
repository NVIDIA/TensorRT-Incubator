load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

td_library(
    name = "TensorRTDialectTdFiles",
    srcs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.td",
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.td",
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td",
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.td",
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.td",
    ],
    includes = ["tensorrt/include"],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTOpsIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.cpp.inc",
        ),
        (
            [
                "-gen-typedef-decls",
                "-dialect=tensorrt",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-dialect=tensorrt",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsTypes.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tensorrt",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tensorrt",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTEnumsIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTAttributesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTInterfacesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTAttrInterfacesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

cc_library(
    name = "TensorRTDialect",
    srcs = [
        "tensorrt/lib/TensorRT/IR/TensorRT.cpp",
        "tensorrt/lib/TensorRT/IR/TypeInferenceInterfaceImpls.cpp",
        "tensorrt/lib/TensorRT/IR/Verification.cpp",
        "tensorrt/lib/TensorRT/IR/EinsumHelper.cpp",
        "tensorrt/lib/TensorRT/IR/EinsumHelper.h",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTAttrInterfacesIncGen",
        ":TensorRTAttributesIncGen",
        ":TensorRTCommonUtils",
        ":TensorRTEnumsIncGen",
        ":TensorRTInterfacesIncGen",
        ":TensorRTOpsIncGen",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TensorRTUtils",
    srcs = [
        "tensorrt/lib/TensorRT/Utils/Utils.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Utils/Utils.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTDialect",
    ],
)

cc_library(
    name = "TensorRTCommonUtils",
    srcs = [
        "tensorrt/lib/Utils/ShapeUtils.cpp",
        "tensorrt/lib/Utils/StaticValueUtils.cpp",
        "tensorrt/lib/Utils/ConstantFoldUtils.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Utils/ShapeUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/StaticValueUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/TensorRTVersion.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/NvInferPluginUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/NvInferAdaptor.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        "@tensorrt10_x86//:tensorrt10",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

td_library(
    name = "TensorKindInterfaceTdFiles",
    srcs = [
        "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.td",
    ],
    includes = ["tensorrt/include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTTensorKindInterfacesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.cpp.inc",
        ),
        (
            ["-gen-attr-interface-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindAttrInterface.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindAttrInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.td",
    deps = [
        ":TensorKindInterfaceTdFiles",
    ],
)

cc_library(
    name = "TensorRTAnalysis",
    srcs = [
        "tensorrt/lib/Analysis/TensorKindAnalysis.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h",
        "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTTensorKindInterfacesIncGen",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BufferizationDialect",
    ],
)

cc_library(
    name = "TensorRTCompilerUtils",
    srcs = [
        "compiler/lib/Utils/RegionUtils.cpp",
    ],
    hdrs = [
        "compiler/include/mlir-tensorrt/Utils/RegionUtils.h",
    ],
    strip_include_prefix = "compiler/include",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TensorRTGenericClustering",
    srcs = [
        "compiler/lib/Transforms/Clustering/Clustering.cpp",
        "compiler/lib/Transforms/Clustering/Patterns.cpp",
    ],
    hdrs = [
        "compiler/include/mlir-tensorrt/Transforms/Clustering/Clustering.h",
        "compiler/include/mlir-tensorrt/Transforms/Clustering/Patterns.h",
    ],
    strip_include_prefix = "compiler/include",
    deps = [
        ":TensorRTCompilerUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "TensorRTTransformsPassesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorRTTransforms",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Transforms/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTTransforms",
    srcs = [
        "tensorrt/lib/TensorRT/Transforms/ApplyBugWorkarounds.cpp",
        "tensorrt/lib/TensorRT/Transforms/BroadcastElimination.cpp",
        "tensorrt/lib/TensorRT/Transforms/ExpandOps.cpp",
        "tensorrt/lib/TensorRT/Transforms/Passes.cpp",
        "tensorrt/lib/TensorRT/Transforms/InferPluginShapes.cpp",
        "tensorrt/lib/TensorRT/Transforms/LegalizeInt8.cpp",
        # "tensorrt/lib/TensorRT/Transforms/RaiseActivations.cpp",
        # "tensorrt/lib/TensorRT/Transforms/RaiseNormalizations.cpp",
        "tensorrt/lib/TensorRT/Transforms/ReshapeElimination.cpp",
        "tensorrt/lib/TensorRT/Transforms/TransposeElimination.cpp",
        # "compiler/lib/Conversion/TensorRTCommon/ConvertToTensorRTCommon.cpp",
    ],
    copts = [
        "-DMLIR_TRT_TARGET_TENSORRT",    
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h",
        # "compiler/include/mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTAnalysis",
        ":TensorRTDialect",
        ":TensorRTGenericClustering",
        ":TensorRTTransformsPassesIncGen",
        ":TensorRTUtils",
        "@tensorrt10_x86//:tensorrt10",
        "@rules_cuda//cuda:runtime",
    ],
)

gentbl_cc_library(
    name = "TensorRTGenericTransformPassesIncGen",
    strip_include_prefix = "compiler/include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MLIRTensorRTGenericTransforms",
            ],
            "compiler/include/mlir-tensorrt/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "compiler/include/mlir-tensorrt/Transforms/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTGeneralTransforms",
    srcs = [
        "compiler/lib/Transforms/DropNestedModules/DropNestedModules.cpp",
        "compiler/lib/Transforms/DuplicateFunctionElimination/DuplicateFunctionElimination.cpp",
        "compiler/lib/Transforms/MemRefCastElimination/MemRefCastElimination.cpp",
        "compiler/lib/Transforms/SCFDetensorizeLoops/SCFDetensorizeLoops.cpp",
    ],
    hdrs = [
        "compiler/include/mlir-tensorrt/Transforms/Passes.h",
        "compiler/include/mlir-tensorrt/Transforms/Transforms.h",
    ],
    strip_include_prefix = "compiler/include",
    deps = [
        ":TensorRTGenericTransformPassesIncGen",
        ":TensorRTAnalysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:TransformDialectTransforms",
    ],
)

gentbl_cc_library(
    name = "TensorRTEnumConverterGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["--gen-tensorrt-enum-converter-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/EnumConverters.inc.cpp",
        ),
    ],
    tblgen = ":mlir-tensorrt-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTEncodingOpInterfaceIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTEncodingOpInterface",
    srcs = [
        "tensorrt/lib/Target/TensorRTEncodingOpInterface/NetworkEncoder.cpp",
        "tensorrt/lib/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/NetworkEncoder.h",
        "tensorrt/include/mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTEncodingOpInterfaceIncGen",
        ":TensorRTEnumConverterGen",
        ":TensorRTUtils",
        ":TensorRTCommonUtils",
        ":TensorRTAnalysis",
        "@rules_cuda//cuda:runtime",
        "@tensorrt10_x86//:tensorrt10",
    ],
)

gentbl_cc_library(
    name = "TensorRTEncodingIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            ["--gen-tensorrt-layer-add-defs"],
            "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.inc.cpp",
        ),
    ],
    tblgen = ":mlir-tensorrt-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
    ],
)

cc_library(
    name = "TensorRTEncodingImpl",
    srcs = [
        "tensorrt/lib/TensorRT/Target/TensorRTEncodingImpl.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTEncodingIncGen",
        ":TensorRTEncodingOpInterface",
    ],
)

cc_library(
    name = "TensorRTDynamicLoader",
    srcs = [
        "executor/lib/Utils/TensorRTDynamicLoader/TensorRTDynamicLoader.cpp",
    ],
    deps = [
        "@rules_cuda//cuda:runtime",
        "@llvm-project//mlir:Support",
        "@tensorrt10_x86//:tensorrt10",
    ],
)

gentbl_cc_library(
    name = "TensorRTTranslationPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorRTTranslation",
            ],
            "include/mlir-tensorrt/Target/TensorRT/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Target/TensorRT/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTTarget",
    srcs = [
        "lib/Target/TensorRT/Registration.cpp",
        "lib/Target/TensorRT/TranslateToTensorRT.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Registration/*.h",
        "include/mlir-tensorrt/Target/TensorRT/*.h",
    ]),
    copts = ["-DMLIR_TRT_TARGET_TENSORRT"],
    strip_include_prefix = "include",
    deps = [
        ":TensorRTDialect",
        # ":TensorRTDynamicLoader",
        ":TensorRTEncodingImpl",
        # ":TensorRTEncodingOpInterface",
        ":TensorRTTranslationPassIncGen",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "TensorRTRegistration",
    srcs = [
        "lib/Registration/Registration.cpp",
    ],
    hdrs = [
        "include/mlir-tensorrt/Conversion/Patterns.h",
    ],
    copts = [
        "-DMLIR_TRT_TARGET_TENSORRT",    
    ],
    strip_include_prefix = "include",
    deps = [
        ":TensorRTGeneralTransforms",
        ":TensorRTTarget",
        ":TensorRTTransforms",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ShapeDialect",
    ],
)

cc_binary(
    name = "mlir-tensorrt-tblgen",
    srcs = ["tools/MlirTensorRtTblgen.cpp"],
    deps = [
        "@llvm-project//mlir:MlirTableGenMain",
    ],
)

cc_binary(
    name = "mlir-tensorrt-opt",
    srcs = ["tools/MlirTensorRtOpt.cpp"],
    copts = [
        "-DMLIR_TRT_TARGET_TENSORRT",
    ],
    deps = [
        ":TensorRTRegistration",
        "@llvm-project//mlir:MlirOptLib",
    ],
)