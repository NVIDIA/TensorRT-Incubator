#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ClusterKindInterface
//===----------------------------------------------------------------------===//

def ClusterKindAttrInterface : AttrInterface<"ClusterKindAttrInterface"> {
    let cppNamespace = "::mlir::plan";
    let methods = [
        InterfaceMethod<
            /*desc=*/"Return the name of the cluster",
            /*retTy-*/"std::string",
            "getClusterKindName",
            /*args=*/(ins ),
            /*body=*/"",
            ""
        >,
        InterfaceMethod<
            /*desc=*/"Return the clustering options for the cluster",
            /*retTy-*/"::mlir::ClusteringOpts",
            "getClusterKindOptions",
            /*args=*/(ins "::mlir::DataFlowSolver &":$solver, "std::optional<int64_t>":$trtMajorVersion),
            /*body=*/"",
            ""
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
            /*desc=*/"Return the pass used to process the cluster kind",
            /*retTy-*/"std::unique_ptr<::mlir::Pass>",
            "getClusterKindPass",
            /*args=*/(ins ),
            /*body=*/"",
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
