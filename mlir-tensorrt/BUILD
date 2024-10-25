load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

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
    name = "TensorRTCommonUtils",
    srcs = [
        "tensorrt/lib/Utils/ConstantFoldUtils.cpp",
        "tensorrt/lib/Utils/NvInferPluginUtils.cpp",
        "tensorrt/lib/Utils/ShapeUtils.cpp",
        "tensorrt/lib/Utils/StaticValueUtils.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/NvInferAdaptor.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/NvInferPluginUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/Options.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/ShapeUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/StaticValueUtils.h",
        "tensorrt/include/mlir-tensorrt-dialect/Utils/TensorRTVersion.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@rules_cuda//cuda:runtime",
        "@tensorrt10_x86//:tensorrt10",
    ],
)

cc_library(
    name = "TensorRTDialect",
    srcs = [
        "tensorrt/lib/TensorRT/IR/EinsumHelper.cpp",
        "tensorrt/lib/TensorRT/IR/EinsumHelper.h",
        "tensorrt/lib/TensorRT/IR/TensorKindOpInterfaceImpl.cpp",
        "tensorrt/lib/TensorRT/IR/TensorRT.cpp",
        "tensorrt/lib/TensorRT/IR/TypeInferenceInterfaceImpls.cpp",
        "tensorrt/lib/TensorRT/IR/Verification.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h",
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTAttrInterfacesIncGen",
        ":TensorRTAttributesIncGen",
        ":TensorRTEnumsIncGen",
        ":TensorRTInterfacesIncGen",
        ":TensorRTOpsIncGen",
        ":TensorRTTensorKindInterfacesIncGen",
        ":TensorRTCommonUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:QuantOps",
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
    name = "TensorRTTensorKindInterfaces",
    srcs = [
        "tensorrt/lib/Interface/TensorKindOpInterface.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTTensorKindInterfacesIncGen",
        "@llvm-project//mlir:Analysis",
    ],
)

cc_library(
    name = "TensorRTAnalysis",
    srcs = [
        "tensorrt/lib/Analysis/TensorKindAnalysis.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTTensorKindInterfaces",
        "@llvm-project//mlir:BufferizationDialect",
    ],
)

gentbl_cc_library(
    name = "TensorRTTransformsPassesIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorRT",
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

gentbl_cc_library(
    name = "TensorRTTransformsActivationsPdllGen",
    strip_include_prefix = "tensorrt/lib/TensorRT/Transforms",
    tbl_outs = [
        (
            ["-x=cpp"],
            "tensorrt/lib/TensorRT/Transforms/RaiseActivations.pdll.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "tensorrt/lib/TensorRT/Transforms/RaiseActivations.pdll",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTTransformsNormalizationsPdllGen",
    strip_include_prefix = "tensorrt/lib/TensorRT/Transforms",
    tbl_outs = [
        (
            ["-x=cpp"],
            "tensorrt/lib/TensorRT/Transforms/RaiseNormalizations.pdll.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "tensorrt/lib/TensorRT/Transforms/RaiseNormalizations.pdll",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
    ],
)

cc_library(
    name = "TensorRTTransforms",
    srcs = [
        "tensorrt/lib/TensorRT/Transforms/ApplyBugWorkarounds.cpp",
        "tensorrt/lib/TensorRT/Transforms/BroadcastElimination.cpp",
        "tensorrt/lib/TensorRT/Transforms/ExpandOps.cpp",
        "tensorrt/lib/TensorRT/Transforms/InferPluginShapes.cpp",
        "tensorrt/lib/TensorRT/Transforms/LegalizeInt8.cpp",
        "tensorrt/lib/TensorRT/Transforms/Passes.cpp",
        "tensorrt/lib/TensorRT/Transforms/RaiseActivations.cpp",
        "tensorrt/lib/TensorRT/Transforms/RaiseNormalizations.cpp",
        "tensorrt/lib/TensorRT/Transforms/ReshapeElimination.cpp",
        "tensorrt/lib/TensorRT/Transforms/TransposeElimination.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h",
    ],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTTransformsActivationsPdllGen",
        ":TensorRTTransformsNormalizationsPdllGen",
        ":TensorRTTransformsPassesIncGen",
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
        ":TensorRTAnalysis",
        ":TensorRTGenericTransformPassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
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
        ":TensorRTTensorKindInterfaces",
        ":TensorRTUtils",
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
        "@llvm-project//mlir:Support",
        "@rules_cuda//cuda:runtime",
        "@tensorrt10_x86//:tensorrt10",
    ],
)

gentbl_cc_library(
    name = "TensorRTTranslationPassIncGen",
    strip_include_prefix = "tensorrt/include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorRTTranslation",
            ],
            "tensorrt/include/mlir-tensorrt-dialect/Target/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tensorrt/include/mlir-tensorrt-dialect/Target/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTTarget",
    srcs = [
        "tensorrt/lib/Target/Registration.cpp",
        "tensorrt/lib/Target/TranslateToTensorRT.cpp",
    ],
    hdrs = [
        "tensorrt/include/mlir-tensorrt-dialect/Target/Passes.h",
        "tensorrt/include/mlir-tensorrt-dialect/Target/TranslateToTensorRT.h",
    ],
    copts = ["-DMLIR_TRT_TARGET_TENSORRT"],
    strip_include_prefix = "tensorrt/include",
    deps = [
        ":TensorRTEncodingImpl",
        ":TensorRTEncodingOpInterface",
        ":TensorRTTranslationPassIncGen",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//mlir:TranslateLib",
    ],
)

gentbl_cc_library(
    name = "TensorRTConversionPassIncGen",
    strip_include_prefix = "compiler/include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MLIRTensorRTConversion",
            ],
            "compiler/include/mlir-tensorrt/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "compiler/include/mlir-tensorrt/Conversion/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTConversion",
    srcs = [
        "compiler/lib/Conversion/TensorRTCommon/ConvertToTensorRTCommon.cpp",
    ],
    hdrs = [
        "compiler/include/mlir-tensorrt/Conversion/Passes.h",    
        "compiler/include/mlir-tensorrt/Conversion/Patterns.h",
        "compiler/include/mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h",
    ],
    strip_include_prefix = "compiler/include",
    deps = [
        ":TensorRTConversionPassIncGen",
        ":TensorRTDialect",
        "@llvm-project//mlir:ConversionPasses",
    ],
)

cc_library(
    name = "TensorRTRegistration",
    srcs = [
        "compiler/lib/Registration/Registration.cpp",
    ],
    hdrs = [
        "compiler/include/mlir-tensorrt/Registration/RegisterMlirTensorRtCoreDialects.h",
        "compiler/include/mlir-tensorrt/Registration/RegisterMlirTensorRtDialects.h",
        "compiler/include/mlir-tensorrt/Registration/RegisterMlirTensorRtPasses.h",
        "compiler/include/mlir-tensorrt/Registration/RegisterMlirTensorRtTranslations.h",
    ],
    copts = ["-DMLIR_TRT_TARGET_TENSORRT"],
    strip_include_prefix = "compiler/include",
    deps = [
        ":TensorRTConversion",
        ":TensorRTGeneralTransforms",
        ":TensorRTTarget",
        ":TensorRTTransforms",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:TransformDialectTransforms",

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
    name = "tensorrt-opt",
    srcs = ["tensorrt/tensorrt-opt/tensorrt-opt.cpp"],
    copts = ["-DMLIR_TRT_TARGET_TENSORRT"],
    linkopts = ["-Wl,-rpath,/opt/src/mlir-tensorrt/bazel-mlir-tensorrt/external/tensorrt10_x86/targets/x86_64-linux-gnu/lib"],
    deps = [
        ":TensorRTRegistration",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

cc_binary(
    name = "mlir-tensorrt-opt",
    srcs = ["tools/MlirTensorRtOpt.cpp"],
    copts = ["-DMLIR_TRT_TARGET_TENSORRT"],
    linkopts = ["-Wl,-rpath,/opt/src/mlir-tensorrt/bazel-mlir-tensorrt/external/tensorrt10_x86/targets/x86_64-linux-gnu/lib"],
    deps = [
        ":TensorRTRegistration",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

buildifier(
    name = "buildifier",
)

# bazel run //:requirements.update
compile_pip_requirements(
    name = "requirements",
    requirements_in = "//:requirements.txt",
    requirements_txt = "//:requirements_lock.txt",
    tags = ["manual"],
)
