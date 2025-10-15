#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORATTRIBUTES_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "mlir-executor/Executor/IR/ExecutorDialect.td"

//===----------------------------------------------------------------------===//
// Executor Dialect Enum Attributes
//===----------------------------------------------------------------------===//

// Enums must be included here since every enum has a corresponding attribute.
include "mlir-executor/Executor/IR/ExecutorEnums.td"

//===----------------------------------------------------------------------===//
// Executor Dialect Attributes Classes and Parameters
//===----------------------------------------------------------------------===//

class Executor_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Executor_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class DefaultValuedEnumParameter<EnumAttrInfo enumInfo, string value>
    : DefaultValuedParameter<enumInfo.cppNamespace # "::" # enumInfo.className,
                    value,
                    "an enum of type " # enumInfo.className> {
  let parser = enumInfo.parameterParser;
  let printer = enumInfo.parameterPrinter;
}

def Executor_DimensionBoundsAttr : Executor_Attr<"DimensionBounds", "dim_bounds"> {
  let description = [{

    An "#executor.dim_bounds" attribute respresents min/max bounds (inclusive) for
    the extent values of shape's dimensions.
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$min,
    "::mlir::DenseI64ArrayAttr":$max
  );

  let assemblyFormat = [{
    `<` `min` `=` $min `,` `max` `=`  $max `>`
  }];

  let genVerifyDecl = 1;
}

def Executor_ValueBoundsAttr : Executor_Attr<"ValueBounds", "value_bounds"> {
  let description = [{

    An `#executor.value_bounds` attribute represents min/max bounds (inclusive) for
    the contents of a tensor.
  }];

  let parameters = (ins
    "::mlir::ElementsAttr":$min,
    "::mlir::ElementsAttr":$max
  );

  let assemblyFormat = [{
    `<` `min` `=` $min `,` `max` `=`  $max `>`
  }];

  let genVerifyDecl = 1;
}

def Executor_ArgumentABIAttr : Executor_Attr<"ArgumentABI", "arg"> {
  let description = [{
    The `#executor.arg` attribute carries information about the ABI of an argument of a function.
  }];

  let parameters = (ins
    EnumParameter<Executor_ArgABIKind>:$abi,
    "::mlir::Type":$value_type
  );

  let assemblyFormat = [{
    `<` $abi `,` $value_type `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "::mlir::executor::ArgABIKind":$abi,
      "::mlir::Type":$value_type), [{
      return Base::get(value_type.getContext(), abi, value_type);
    }]>
  ];
}

def Executor_FunctionMetadataAttr : Executor_Attr<"FunctionMetadata", "func_meta"> {
  let description = [{
    The `#executor.func_meta` attribute carries information about the
    function signature and important semantic information. It should be created
    before `memref` types are eliminated (which either lower to unpacked scalars
    or to structs/tables, both of which drop information about what the arguments
    actually information).

    The `num_output_args` parameter describes which of the `args` parameters are output
    parameters (e.g. "output buffers" for functions in destination-passing-style).
    The last `num_output_args` should be interpreted as output parameters.

    The `arg_bounds|result_bounds` arrays should contain one Attribute for each
    element in the corresponding Type array. Attributes may be either
    `DimensionBoundsAttr`, `ValueBoundsAttr` or `UnitAttr`.

    `UnitAttr` is expected for any type that is not a scalar or memref type.
    `DimensionBoundsAttr` may only be used for memref types. `ValueBoundsAttr` may be
    used for staticly shaped memref types or scalar types.
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"Type">:$args,
    OptionalArrayRefParameter<"Type">:$results,
    "int64_t":$num_output_args,
    OptionalArrayRefParameter<"::mlir::Attribute">:$arg_bounds,
    OptionalArrayRefParameter<"::mlir::Attribute">:$result_bounds,
    OptionalParameter<"FlatSymbolRefAttr">:$shape_func,
    DefaultValuedEnumParameter<Executor_CallingConvention,
      "CallingConvention::unpacked">:$cconv
  );

  let assemblyFormat = [{
    `<`
    custom<TypesWithBoundsAttrs>($args, $arg_bounds) `,`
    custom<TypesWithBoundsAttrs>($results, $result_bounds) `,`
    struct($num_output_args, $shape_func, $cconv)  `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<Type>":$args, "ArrayRef<Type>":$results,
        "int64_t":$num_output_args,
        CArg<"ArrayRef<Attribute>", "ArrayRef<Attribute>{}">:$arg_bounds,
        CArg<"ArrayRef<Attribute>", "ArrayRef<Attribute>{}">:$result_bounds), [{
      return Base::get($_ctxt, args, results, num_output_args, arg_bounds,
        result_bounds, FlatSymbolRefAttr{}, CallingConvention::unpacked);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Verify that the attribute is valid when attached to the given operation.
    /// If not, then emits an error using the provided function.
    ::mlir::LogicalResult verifyAgainstFunc(func::FuncOp func) const;
  }];
}

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORATTRIBUTES_TD
