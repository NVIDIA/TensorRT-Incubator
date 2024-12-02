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
              "::mlir::Operation *":$op,
              "::mlir::DataFlowSolver &":$solver),
            /*body=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/"Return the region outlining options for the cluster",
            /*retTy*/"std::optional<OutlineRegionOptions>",
            "getClusterOutliningOptions",
            (ins "::mlir::MLIRContext*":$context, "::mlir::SymbolTable&":$moduleSymbolTable),
            "",
            ""
        >,
        InterfaceMethod<
            "Returns the cluster benefit",
            "int64_t",
            "getClusterBenefit",
            (ins),
            "",
            ""
        >,
        InterfaceMethod<
            "Return a post-clustering filter function",
            "std::function<bool(const Cluster &cluster)>",
            "getClusterFilter",
            (ins),
            "",
            ""
        >
    ];
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
