#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTATTRIBUTES
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.td"

//===----------------------------------------------------------------------===//
// TensorRT Dialect Enums
//===----------------------------------------------------------------------===//

// Enums must be included here since every enum has a corresponding attribute.
include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td"

//===----------------------------------------------------------------------===//
// TensorRT Dialect Attributes Classes and Parameters
//===----------------------------------------------------------------------===//

class TensorRT_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorRT_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// The regular "ArrayRefParameter" won't parse empty lists. This
// variant parses/prints a dimension list as equivalent to
// ```
// `[` ( $dims^ )? `]`
// ```
def TensorRT_DimList : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "mlir::tensorrt::parseDimList($_parser)";
  let printer = "mlir::tensorrt::printDimList($_printer, $_self)";
}

//===----------------------------------------------------------------------===//
// ShapeProfileAttr
//===----------------------------------------------------------------------===//

def TensorRT_ShapeProfileAttr : TensorRT_Attr<"ShapeProfile", "shape_profile"> {
  let summary = "An Attribute representing the min, max, and optimum shapes for a Tensor type.";

  let description = [{
    The `#tensorrt.shape_profile` attribute describes the bounds on the shape of a tensor
    type (usually the input to a function).

    The bounds are described by `min`, `opt`, and `max`. For each dimension
    `i` the bounds must satisfy the constraints `min[i] <= opt[i] <= max[i]`.

    No element of the bounds can be unknown.
  }];

  let parameters = (ins
    TensorRT_DimList:$min,
    TensorRT_DimList:$opt,
    TensorRT_DimList:$max
  );

  let builders = [
    // Construct
    AttrBuilder<(ins "::mlir::tensorrt::DynamicDimensionBounds":$batchSizeBounds, "ArrayRef<int64_t>":$nonBatchDims)>,
    AttrBuilderWithInferredContext<(ins "::mlir::TensorType":$staticType), [{
      if(!staticType.hasStaticShape())
        return failure();
      ArrayRef<int64_t> shape = staticType.getShape();
      return $_get(staticType.getContext(), shape, shape, shape);
    }], "FailureOr<ShapeProfileAttr>">
  ];

  let assemblyFormat = [{
    `<` `min` `=` $min `,`
        `opt` `=` $opt `,`
        `max` `=` $max
    `>`
  }];

  let genVerifyDecl = 1;
}


#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTATTRIBUTES
