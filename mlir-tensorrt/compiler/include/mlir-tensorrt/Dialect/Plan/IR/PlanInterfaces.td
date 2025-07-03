#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorBoundsAttrInterface
//===----------------------------------------------------------------------===//

def TensorBoundsAttrInterface : AttrInterface<"TensorBoundsAttrInterface"> {
  let cppNamespace = "::mlir::plan";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the shape bounds associated with the attribute",
      /*retTy=*/"LogicalResult",
      "getShapeBounds",
      (ins "llvm::SmallVectorImpl<int64_t> &":$min,
           "llvm::SmallVectorImpl<int64_t> &":$max),
      /*body=*/"",
      ""
    >,
    InterfaceMethod<
      /*desc=*/"Return the integer value bounds associated with the attribute",
      /*retTy=*/"LogicalResult",
      "getIntegerValueBounds",
      (ins "llvm::SmallVectorImpl<llvm::APInt> &":$min,
           "llvm::SmallVectorImpl<llvm::APInt> &":$max),
      /*body=*/"",
      ""
    >
  ];
}

//===----------------------------------------------------------------------===//
// ClusterKindInterface
//===----------------------------------------------------------------------===//

def ClusterKindAttrInterface : AttrInterface<"ClusterKindAttrInterface"> {
    let cppNamespace = "::mlir::plan";
    let methods = [
      InterfaceMethod<
          /*desc=*/[{
            Return true if the cluster kind supports the given input kind.
          }],
          "bool",
          "supportsInputKind",
          (ins "::mlir::plan::InputKind":$input_kind),
          "",
          ""
        >,
        InterfaceMethod<
          /*desc=*/[{
            Returns the clustering options for this cluster kind. The
            scope of the clustering is restricted to `op`, so `op` can
            be utilized to perform e.g. analysis conversion and use
            the results in the callbacks returned from this method.

            The `solver` is provided in order to query the results of
            SparseConstantPropagation and TensorKindAnalysis.
          }],
          /*retTy-*/"FailureOr<::mlir::ClusteringOpts>",
          "getClusterKindOptions",
          /*args=*/(ins
            "::mlir::plan::InputKind":$input_kind,
            "::mlir::Operation *":$op,
            "::mlir::DataFlowSolver &":$solver),
          /*body=*/"",
          /*defaultImplementation=*/""
        >,
        InterfaceMethod<
          /*desc*/[{
            Returns the default memory space used for this cluster kind.
          }],
          /*retTy*/"::mlir::plan::MemorySpace",
          "getDefaultMemorySpace",
          (ins),
          "",
          "return ::mlir::plan::MemorySpace::device;"
        >,
        InterfaceMethod<
          /*desc=*/[{
            Return true if the cluster requires closure prior to
            outlining. If false, then the the outlining step must accept
            a `plan::InlineGroupOp`.

            Generally, this can be set to return true if the cluster kind
            does not require value or dimension bounds information for the
            cluster. Bounds for values and dynamic shapes are only analyzed
            during the region closure step.
          }],
          /*retTy*/"bool", "requiresClosure",
          (ins "::mlir::plan::InputKind":$input_kind),
          ""
        >,
        InterfaceMethod<
          /*desc=*/"Return the region outlining options for the cluster",
          /*retTy*/"std::optional<OutlineRegionOptions>",
          "getClusterOutliningOptions",
          (ins
            "::mlir::plan::InputKind":$input_kind,
            "::mlir::MLIRContext*":$context,
            "::mlir::SymbolTable&":$moduleSymbolTable),
          "",
          ""
        >,
        InterfaceMethod<
          /*desc=*/[{
            This method specifies a method for manually outlining a closed
            cluster, which is either a `plan::InlineClosedGroupOp` or
            a `plan::InlineClosedAllocGroupOp`.

            The `moduleSymbolTable` is the symbol table of the immediately
            enclosing module.
          }],
          "LogicalResult",
          "outlineClosedCluster",
          (ins
            "::mlir::plan::InputKind":$input_kind,
            "::mlir::RewriterBase&":$rewriter,
            "::mlir::Operation*":$op,
            "::mlir::SymbolTable&":$moduleSymbolTable),
          "",
          "return failure();"
        >,
        InterfaceMethod<
          "Returns the cluster benefit",
          "int64_t",
          "getClusterBenefit",
          (ins "::mlir::plan::InputKind":$input_kind),
          "",
          ""
        >,
        InterfaceMethod<
          "Return a post-clustering filter function",
          "std::function<bool(const Cluster &cluster)>",
          "getClusterFilter",
          (ins "::mlir::plan::InputKind":$input_kind),
          "",
          ""
        >
    ];
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
