#ifndef MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE

include "mlir/IR/OpBase.td"

def BoundsAttrInterface : AttrInterface<"BoundsAttrInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    This interface should be used by attributes which describe
    static bounds associated with a particular value.
    Typical use-cases include describing the bounds of a function argument.

    Different kinds of bounds can be described: bounds for
    the shape of a SSA value with a ShapedType, bounds for
    the values of an SSA value with a ShapedType, or bounds
    for the value of a scalar typed SSA value.

    In the case of bounds of the dimension extents of
    a ShapedType-typed value, the bounds are described as
    flat list of minimum and maximum values for each dimension.

    In the case of bounds of the values of a ShapedType-typed value,
    the bounds are described using an ElementsAttr for the minimum
    and maximum values.

    In the case of bounds of a scalar typed SSA value, the bounds
    are described using APInt or APFloat values for the minimum
    and maximum values.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the shape bounds associated with the attribute",
      /*retTy=*/"::mlir::LogicalResult",
      "getShapeRange",
      (ins "llvm::SmallVectorImpl<int64_t> &":$min,
           "llvm::SmallVectorImpl<int64_t> &":$max),
      /*body=*/"",
      "return ::mlir::failure();"
    >,
    InterfaceMethod<
      /*desc=*/"Return the shape bounds associated with the attribute",
      /*retTy=*/"std::optional<::mlir::ConstantIntRanges>",
      "getDimensionRange",
      (ins "int64_t":$dimension),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"Return the integer value bounds associated with the attribute",
      /*retTy=*/"std::optional<::mlir::ConstantIntRanges>",
      "getIntegerValueRange",
      (ins ),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"For bounds describing shaped types, return the integer value "
      "bounds associated with the attribute.",
      /*retTy=*/"std::optional<::mlir::ConstantIntRanges>",
      "getIntegerValueRangeAtCoordinate",
      (ins "::mlir::ArrayRef<uint64_t>":$coordinate),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"Return the integer value bounds associated with the attribute",
      /*retTy=*/"std::optional<::llvm::SmallVector<::mlir::ConstantIntRanges>>",
      "getIntegerValueRanges",
      (ins ),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"Return the scalar float value bounds associated with the attribute",
      /*retTy=*/"std::optional<::mlir::FloatValueRange>",
      "getFloatValueRange",
      (ins ),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"For bounds describing shaped types, return the float value "
               "bounds associated with the attribute.",
      /*retTy=*/"std::optional<::llvm::SmallVector<::mlir::FloatValueRange>>",
      "getFloatValueRanges",
      (ins ),
      /*body=*/"",
      "return std::nullopt;"
    >,
    InterfaceMethod<
      /*desc=*/"For bounds describing shaped types, returns the float value "
               "bounds at a particular coordinate.",
      /*retTy=*/"std::optional<::mlir::FloatValueRange>",
      "getFloatValueRangeAtCoordinate",
      (ins "::mlir::ArrayRef<uint64_t>":$coordinate),
      /*body=*/"",
      "return std::nullopt;"
    >
  ];
}

#endif // MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE
