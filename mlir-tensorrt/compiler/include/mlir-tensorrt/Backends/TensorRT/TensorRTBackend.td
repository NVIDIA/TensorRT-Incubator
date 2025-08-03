#ifndef MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND
#define MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND

include "mlir-tensorrt/Dialect/Plan/IR/PlanDialect.td"
include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.td"

def Plan_TensorRTBackendAttr : Plan_Attr<"TensorRTBackend", "tensorrt_backend",
      [DeclareAttrInterfaceMethods<CompilerBackendAttrInterface,
        ["getClusterFilter", "outlineClosedCluster"]>]> {
  let parameters = (ins
    "bool":$disallow_shape_tensor_calculations,
    "int64_t":$benefit,
    OptionalParameter<"std::optional<int64_t>">:$tensorrt_major_version);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND
