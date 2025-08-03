#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANATTRIBUTES
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANATTRIBUTES

include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.td"
include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.td"
include "mlir-tensorrt-common/Interfaces/BoundsAttrInterface.td"

def Plan_BoundsAttr : Plan_Attr<"Bounds", "bounds", [
    DeclareAttrInterfaceMethods<BoundsAttrInterface, [
      "getShapeRange",
      "getDimensionRange",
      "getIntegerValueRange",
      "getIntegerValueRangeAtCoordinate",
      "getIntegerValueRanges",
      "getFloatValueRange",
      "getFloatValueRanges",
      "getFloatValueRangeAtCoordinate"
    ]>]>{
  let parameters = (ins
      EnumParameter<Plan_BoundsKind>:$kind,
      OptionalParameter<"DenseI64ArrayAttr">:$min_shape,
      OptionalParameter<"DenseI64ArrayAttr">:$max_shape,
      OptionalParameter<"ElementsAttr">:$min_values,
      OptionalParameter<"ElementsAttr">:$max_values);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    // Note: we use SmallVectorImpl instead of ArrayRef to avoid
    // having to deal with the template resolution issues from
    // the Attribute base class.
    AttrBuilder<(ins "BoundsKind":$kind,
      "ArrayRef<int64_t>":$min,
      "ArrayRef<int64_t>":$max)>,
    // Construct a None (empty) bounds.
    AttrBuilder<(ins )>
  ];

  let extraClassDeclaration = [{
    /// Returns true if this bounds is for shape dimension extents.
    bool isShapeBound() const {
      return getKind() == BoundsKind::Shape;
    }

    /// Returns true if this bounds is a 'none' bounds kind.
    bool isNone() const {
      return getKind() == BoundsKind::None;
    }

    /// Returns true if this bounds is for values of a tensor.
    bool isValueBound() const {
      return getKind() == BoundsKind::Value;
    }

    /// Returns the type of the values bounds attributes. Asserts
    /// that the bounds has 'value' kind.
    RankedTensorType getValuesType() const {
      assert(isValueBound() && "expected value bound kind");
      return llvm::cast<RankedTensorType>(getMaxValues().getType());
    }

  }];
}

def BoundsAttrArray : TypedArrayAttrBase<Plan_BoundsAttr,
    "BoundsAttr array attribute">;

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANATTRIBUTES
