#ifndef MLIR_TENSORRT_BACKENDS_HOST_HOSTBACKEND
#define MLIR_TENSORRT_BACKENDS_HOST_HOSTBACKEND

include "mlir-tensorrt/Dialect/Plan/IR/PlanDialect.td"
include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.td"

def Plan_HostClusterKindAttr : Plan_Attr<"HostClusterKind", "host_cluster",
      [DeclareAttrInterfaceMethods<ClusterKindAttrInterface>]> {
  let parameters = (ins "int64_t":$benefit);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // MLIR_TENSORRT_BACKENDS_HOST_HOSTBACKEND
