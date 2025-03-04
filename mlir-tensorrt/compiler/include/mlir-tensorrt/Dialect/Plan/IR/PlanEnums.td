#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANENUMS
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANENUMS

include "mlir-tensorrt/Dialect/Plan/IR/PlanDialect.td"
include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.td"

include "mlir/IR/EnumAttr.td"

class Plan_I32EnumAttr<string name, string description,
  list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::plan";
  let genSpecializedAttr = 0;
}

class Plan_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic, list<Trait> traits = []> :
  EnumAttr<Plan_Dialect, intEnumAttr, mnemonic, traits> {
  let assemblyFormat = "`<` $value `>`";
}

def Plan_BoundsKind : Plan_I32EnumAttr<
  "BoundsKind",
  "kinds of bounds",
  [
    I32EnumAttrCase<"Shape", 0, "shape">,
    I32EnumAttrCase<"Value", 1, "value">,
    I32EnumAttrCase<"None",  2, "none">
  ]
>;
def Plan_BoundsKindAttr : Plan_EnumAttr<Plan_BoundsKind, "bounds_kind">;


def Plan_DeviceKind : Plan_I32EnumAttr<
  "DeviceKind",
  "kind of device in computing system",
  [
    I32EnumAttrCase<"CPU", 0, "CPU">,
    I32EnumAttrCase<"GPU", 1, "GPU">
  ]
>;

class Plan_MemorySpaceCase<string name, int value> : I32EnumAttrCase<name, value>;
def Plan_MemorySpaceUnknown : Plan_MemorySpaceCase<"unknown", 0>;
def Plan_MemorySpaceHost : Plan_MemorySpaceCase<"host", 1>;
def Plan_MemorySpaceHostPinned : Plan_MemorySpaceCase<"host_pinned", 2>;
def Plan_MemorySpaceDevice : Plan_MemorySpaceCase<"device", 3>;
def Plan_MemorySpaceUnified : Plan_MemorySpaceCase<"unified", 4>;

def Plan_MemorySpace : Plan_I32EnumAttr<
  "MemorySpace",
  "",
  [
    Plan_MemorySpaceUnknown,
    Plan_MemorySpaceHost,
    Plan_MemorySpaceHostPinned,
    Plan_MemorySpaceDevice,
    Plan_MemorySpaceUnified
  ]
>;

def Plan_MemorySpaceAttr : Plan_EnumAttr<Plan_MemorySpace, "memory_space",
      [DeclareAttrInterfaceMethods<TensorKindAttrInterface,
                                  ["getTensorKind"]>]> {
  let extraClassDeclaration = [{
    /// Returns true if the memory space can be directly read/written on the host.
    bool isHostVisible() const;

    /// Returns true if the memory space can be directly read/written from the host's
    /// associated NVIDIA GPUs (does not distinguish between particular GPUs).
    bool isGpuVisible() const;

    /// Returns true if the memory space can be directly read/written from the host's
    /// associated device of specified kind.
    bool isVisible(plan::DeviceKind deviceKind) const;
  }];
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANENUMS
