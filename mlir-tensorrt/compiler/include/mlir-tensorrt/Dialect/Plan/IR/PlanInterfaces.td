#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CompilerBackendAttrInterface
//===----------------------------------------------------------------------===//

def CompilerBackendAttrInterface : AttrInterface<"CompilerBackendAttrInterface"> {
    let cppNamespace = "::mlir::plan";
    let methods =
        [InterfaceMethod<
             /*desc=*/[{
            Return true if the backend supports the given input kind.
          }],
             "bool", "supportsInputKind",
             (ins "::mlir::plan::InputKind":$input_kind), "", "">,
         InterfaceMethod<
             /*desc=*/[{
            Returns the clustering options for this backend. The
            scope of the clustering is restricted to `op`, so `op` can
            be utilized to perform e.g. analysis conversion and use
            the results in the callbacks returned from this method.

            The `solver` is provided in order to query the results of
            SparseConstantPropagation and TensorKindAnalysis.
          }],
             /*retTy-*/ "FailureOr<::mlir::ClusteringOpts>",
             "getClusterKindOptions",
             /*args=*/
             (ins "::mlir::plan::InputKind":$input_kind,
                 "::mlir::Operation *":$op, "::mlir::DataFlowSolver &":$solver),
             /*body=*/"",
             /*defaultImplementation=*/"">,
         InterfaceMethod<
             /*desc*/ [{
            Returns the default memory space used for this backend.
          }],
             /*retTy*/ "::mlir::plan::MemorySpace", "getDefaultMemorySpace",
             (ins), "", "return ::mlir::plan::MemorySpace::device;">,
         InterfaceMethod<
             /*desc=*/[{
            Returns true if the backend supports lowering this cluster
            using a destination-passing-style (DPS) calling convention
            where the results are pre-allocated.
          }],
             /*retTy*/ "bool", "supportsDestinationStyleCallingConvention",
             (ins "::mlir::Operation*":$inlineGroupOp), "", [{
            return false;
          }]>,
         InterfaceMethod<
             /*desc=*/[{
            Returns true if the backend recommends lowering this cluster
            using a destination-passing-style (DPS) calling convention
            where the results are pre-allocated.

            This is only supported when the cluster results are all
            `tensor` typed. If any of the tensor types have dynamic shape,
            then shape analysis will occur at the region closure step
            in order to derive the exact shape of the result tensors
            that must be allocated prior to the region op and passed
            as destination arguments. If this analysis fails, then the
            compiler may ignore this recommendation and lower the cluster
            using a non-DPS calling convention.
          }],
             /*retTy*/ "bool", "preferDestinationStyleCallingConvention",
             (ins "::mlir::Operation*":$inlineGroupOp), "", [{
            return false;
          }]>,
         InterfaceMethod<
             /*desc=*/[{
            Returns true if the producer should be cloned into the
            region or passed by argument during the region closure step.
          }],
             /*retTy*/ "bool", "shouldCloneProducer",
             (ins "::mlir::Operation*":$inlineGroupOp,
                 "::mlir::Operation*":$producer),
             "", [{
            return ::mlir::plan::detail::shouldCloneProducerDefault(producer,
              inlineGroupOp->getRegions().front());
          }]>,
         InterfaceMethod<
             /*desc=*/"Return the region outlining options for the backend",
             /*retTy*/ "std::optional<OutlineRegionOptions>",
             "getClusterOutliningOptions",
             (ins "::mlir::plan::InputKind":$input_kind,
                 "::mlir::MLIRContext*":$context,
                 "::mlir::SymbolTable&":$moduleSymbolTable),
             "", "">,
         InterfaceMethod<
             /*desc=*/[{
            This method specifies a method for manually outlining a closed
            backend, which is either a `plan::DpsClusterOp` or
            a `plan::AllocClusterOp`.

            The `moduleSymbolTable` is the symbol table of the immediately
            enclosing module.
          }],
             "LogicalResult", "outlineClosedCluster",
             (ins "::mlir::plan::InputKind":$input_kind,
                 "::mlir::RewriterBase&":$rewriter, "::mlir::Operation*":$op,
                 "::mlir::SymbolTable&":$moduleSymbolTable),
             "", "return failure();">,
         InterfaceMethod<"Returns the backend benefit", "int64_t",
                         "getClusterBenefit",
                         (ins "::mlir::plan::InputKind":$input_kind), "", "">,
         InterfaceMethod<"Return a post-s filter function",
                         "std::function<bool(const Cluster &cluster)>",
                         "getClusterFilter",
                         (ins "::mlir::plan::InputKind":$input_kind), "", "">,
         InterfaceMethod<
             /*desc=*/[{
            Returns true if the backend requires shape bounds for dynamically
            shaped input tensors. The `useDestinationStyleCallingConvention`
            parameter indicates whether the cluster will be called using
            destination-passing style (caller-allocated results) or alloc-style
            (callee-allocated results).
          }],
             /*retTy*/ "bool", "requiresInputBoundsForDynamicShapes",
             (ins "bool":$useDestinationStyleCallingConvention),
             "", [{
            return false;
          }]>,
         InterfaceMethod<
             /*desc=*/[{
            Returns true if the backend requires shape bounds for dynamically
            shaped output tensors. The `useDestinationStyleCallingConvention`
            parameter indicates whether the cluster will be called using
            destination-passing style (caller-allocated results) or alloc-style
            (callee-allocated results).
          }],
             /*retTy*/ "bool", "requiresOutputBoundsForDynamicShapes",
             (ins "bool":$useDestinationStyleCallingConvention),
             "", [{
            return false;
          }]>];
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
