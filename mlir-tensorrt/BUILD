load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

td_library(
    name = "TensorRTDialectTdFiles",
    srcs = [
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttributes.td",
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTDialect.td",
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTEnums.td",
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTInterfaces.td",
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOps.td",
        "include/mlir-tensorrt/Dialect/TensorRT/IR/generated/TensorRT.8.4.3.1.Ops.td",
    ],
    includes = ["include"],
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
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOps.cpp.inc",
        ),
        (
            [
                "-gen-typedef-decls",
                "-dialect=tensorrt",
            ],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOpsTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-dialect=tensorrt",
            ],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOpsTypes.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tensorrt",
            ],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tensorrt",
            ],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOpsDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOps.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTEnumsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTEnums.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTAttributesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttributes.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTInterfaces.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTAttrInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTInterfaces.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

cc_library(
    name = "TensorRTUtils",
    srcs = [
        "lib/Dialect/TensorRT/Utils/Utils.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Dialect/TensorRT/Utils/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":TensorRTDialect",
    ],
)

cc_library(
    name = "TensorRTCommonUtils",
    srcs = [
        "lib/Utils/EinsumHelper.cpp",
        "lib/Utils/IndexUtils.cpp",
        "lib/Utils/PassTreePrinter.cpp",
        "lib/Utils/RegionUtils.cpp",
        "lib/Utils/ShapeUtils.cpp",
        "lib/Utils/StaticValueUtils.cpp",
        "lib/Utils/TensorUtils.cpp",
        "lib/Utils/VectorUtils.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Utils/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TensorRTDialect",
    srcs = [
        "lib/Dialect/TensorRT/IR/TensorRT.cpp",
        "lib/Dialect/TensorRT/IR/TypeInferenceInterfaceImpls.cpp",
        "lib/Dialect/TensorRT/IR/Verification.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTDialect.h",
        "include/mlir-tensorrt/Utils/*.h",
    ]),
    strip_include_prefix = "include",
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
    name = "TensorRTAnalysis",
    srcs = [
        "lib/Dialect/TensorRT/Analysis/ShapeTensorBoundsAnalysis.cpp",
        "lib/Dialect/TensorRT/Analysis/TensorKindAnalysis.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Dialect/TensorRT/Analysis/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":TensorRTDialect",
        "@llvm-project//mlir:Analysis",
    ],
)

gentbl_cc_library(
    name = "TensorRTTransformsPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorRTTransforms",
            ],
            "include/mlir-tensorrt/Dialect/TensorRT/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/Transforms/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTGenericClustering",
    srcs = [
        "lib/Transforms/Clustering/Clustering.cpp",
        "lib/Transforms/Clustering/Patterns.cpp",
    ],
    hdrs = [
        "include/mlir-tensorrt/Transforms/Clustering/Clustering.h",
        "include/mlir-tensorrt/Transforms/Clustering/Patterns.h",
    ],
    strip_include_prefix = "include",
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TensorRTTransforms",
    srcs = [
        "lib/Dialect/TensorRT/Transforms/ApplyBugWorkarounds.cpp",
        "lib/Dialect/TensorRT/Transforms/BroadcastElimination.cpp",
        "lib/Dialect/TensorRT/Transforms/ExpandOps.cpp",
        "lib/Dialect/TensorRT/Transforms/FuncCompositionReport.cpp",
        "lib/Dialect/TensorRT/Transforms/LegalizeInt8.cpp",
        "lib/Dialect/TensorRT/Transforms/LowerLoopsToSCF.cpp",
        "lib/Dialect/TensorRT/Transforms/SegmentTensorRTOps.cpp",
        "lib/Dialect/TensorRT/Transforms/SetShapeProfiles.cpp",
        "lib/Dialect/TensorRT/Transforms/SinkTensorRTConstantOps.cpp",
        "lib/Dialect/TensorRT/Transforms/TensorRTRegionOutlining.cpp",
        "lib/Dialect/TensorRT/Transforms/TransposeElimination.cpp",
        "lib/Conversion/TensorRTCommon/ConvertToTensorRTCommon.cpp",
    ],
    copts = [
        "-DCRUISE_EXPERIMENTAL_BAZEL_BUILD",
        "-DMLIR_TRT_TARGET_TENSORRT",    
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Dialect/TensorRT/Transforms/*.h",
        "include/mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":TensorRTAnalysis",
        ":TensorRTDialect",
        ":TensorRTGenericClustering",
        ":TensorRTTransformsPassesIncGen",
        ":TensorRTUtils",
    ],
)

cc_library(
    name = "TensorRTPipelines",
    srcs = [
        "lib/Pipelines/TensorRTPassPipelines.cpp",
    ],
    hdrs = [
        "include/mlir-tensorrt/Pipelines/Pipelines.h",
    ],
    copts = ["-DCRUISE_EXPERIMENTAL_BAZEL_BUILD"],
    strip_include_prefix = "include",
    deps = [
        ":TensorRTTransforms",
    ],
)

gentbl_cc_library(
    name = "TensorRTGenericTransformPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MLIRTensorRTGenericTransforms",
            ],
            "include/mlir-tensorrt/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Transforms/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTGeneralTransforms",
    srcs = [
        "lib/Transforms/DropNestedModules/DropNestedModules.cpp",
        "lib/Transforms/EliminateI1Tensors/EliminateI1Tensors.cpp",
        "lib/Transforms/TransformDialectInterpreter/TransformDialectInterpreter.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Transforms/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":TensorRTGenericTransformPassesIncGen",
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
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-tensorrt-enum-converter-defs"],
            "include/mlir-tensorrt/Target/TensorRT/TensorRTEncodingOpInterface/EnumConverters.inc.cpp",
        ),
    ],
    tblgen = ":mlir-tensorrt-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTEnums.td",
    deps = [
        ":TensorRTDialectTdFiles",
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorRTEncodingOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-tensorrt/Target/TensorRT/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-tensorrt/Target/TensorRT/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tensorrt/Target/TensorRT/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "TensorRTEncodingOpInterface",
    srcs = [
        "lib/Target/TensorRT/TensorRTEncodingOpInterface/NetworkEncoder.cpp",
        "lib/Target/TensorRT/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.cpp",
    ],
    hdrs = glob([
        "include/mlir-tensorrt/Target/TensorRT/TensorRTEncodingOpInterface/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTEncodingOpInterfaceIncGen",
        ":TensorRTEnumConverterGen",
        ":TensorRTUtils",
        "@cuda",
        "@tensorrt8601",
    ],
    # This is needed for CUDA 12 capability, as the vendored TensorRT is built with CUDA 11
    # TODO[MLA-15279]: Remove once we vendor a CUDA 12 version of TRT 8.6
    data = [
        "@cuda112//:cublas",
        "@cuda112//:cublaslt",
    ],
    linkopts = ["-Wl,-rpath,./external/cuda112/targets/x86_64-linux/lib"],
)

gentbl_cc_library(
    name = "TensorRTEncodingIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-tensorrt-layer-add-defs"],
            "include/mlir-tensorrt/Dialect/TensorRT/Target/TensorRTEncodingImpl.inc.cpp",
        ),
    ],
    tblgen = ":mlir-tensorrt-tblgen",
    td_file = "include/mlir-tensorrt/Dialect/TensorRT/IR/TensorRTOps.td",
    deps = [
        ":TensorRTDialectTdFiles",
    ],
)

cc_library(
    name = "TensorRTEncodingImpl",
    srcs = [
        "lib/Dialect/TensorRT/Target/TensorRTEncodingImpl.cpp",
    ],
    hdrs = [
        "include/mlir-tensorrt/Dialect/TensorRT/Target/TensorRTEncodingImpl.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TensorRTDialect",
        ":TensorRTEncodingIncGen",
        ":TensorRTEncodingOpInterface",
    ],
)

cc_library(
    name = "TensorRTDynamicLoader",
    srcs = [
        "lib/Utils/TensorRTDynamicLoader/TensorRTDynamicLoader.cpp",
    ],
    deps = [
        "@cuda",
        "@llvm-project//mlir:Support",
        "@tensorrt8601",
    ],
    # This is needed for CUDA 12 capability, as the vendored TensorRT is built with CUDA 11
    # TODO[MLA-15279]: Remove once we vendor a CUDA 12 version of TRT 8.6
    data = [
        "@cuda112//:cublas",
        "@cuda112//:cublaslt",
    ],
    linkopts = ["-Wl,-rpath,./external/cuda112/targets/x86_64-linux/lib"],
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
        ":TensorRTDynamicLoader",
        ":TensorRTEncodingImpl",
        ":TensorRTEncodingOpInterface",
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
        "-DCRUISE_EXPERIMENTAL_BAZEL_BUILD",
        "-DMLIR_TRT_TARGET_TENSORRT",    
    ],
    strip_include_prefix = "include",
    deps = [
        ":TensorRTGeneralTransforms",
        ":TensorRTPipelines",
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
        "-DCRUISE_EXPERIMENTAL_BAZEL_BUILD",
        "-DMLIR_TRT_TARGET_TENSORRT",
    ],
    deps = [
        ":TensorRTRegistration",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
