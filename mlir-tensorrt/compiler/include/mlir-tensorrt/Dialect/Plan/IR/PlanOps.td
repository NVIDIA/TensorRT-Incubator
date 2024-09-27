#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANOPS_TD
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANOPS_TD

include "mlir-tensorrt/Dialect/Plan/IR/PlanDialect.td"
include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.td"
include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"

class Plan_NativeOpTrait<string name,
        list<Trait> traits = [],
        code extraOpDeclaration = "",
        code extraOpDefinition = "">
      : NativeOpTrait<name, traits, extraOpDeclaration, extraOpDefinition> {
  let cppNamespace = "::mlir::plan::PlanOpTrait";
}

def PlanDialectOpTrait : Plan_NativeOpTrait<"PlanDialectOp">;

class Plan_Op<string mnemonic, list<Trait> traits = []>
    : Op<Plan_Dialect, mnemonic, !listconcat(traits, [PlanDialectOpTrait])>;

//===----------------------------------------------------------------------===//
// GroupOpBase
//===----------------------------------------------------------------------===//

class Plan_GroupOpBase<string mnemonic, list<Trait> traits = []>
  : Plan_Op<mnemonic, !listconcat(traits, [
      RecursiveMemoryEffects
    ])> {
  let regions = (region SizedRegion<1>:$body);

  code baseExtraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // Convenience Methods
    //===------------------------------------------------------------------===//

    /// Return the value yielded from the body corresponding to the i-th result.
    Value getYieldedValueForResult(unsigned idx) {
      assert(idx < (*this)->getNumResults() && "result index is out-of-bounds");
      return getBody().front().getTerminator()->getOperand(idx);
    }

    /// Return the body Block.
    Block *getBodyBlock() {
      assert(getBody().getBlocks().size() == 1 && "expected single-block region");
      return &getBody().front();
    }

    /// Return the terminator of the body block.
    plan::YieldOp getYield();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "plan"; }
  }];

  let extraClassDefinition = [{
    plan::YieldOp $cppClass::getYield() {
      return cast<plan::YieldOp>(getBodyBlock()->getTerminator());
    }
  }];

  let extraClassDeclaration = baseExtraClassDeclaration;
}


//===----------------------------------------------------------------------===//
// InlineGroupOp
//===----------------------------------------------------------------------===//

def Plan_InlineGroupOp : Plan_GroupOpBase<"inline_group", [
  NoRegionArguments,
  OpAsmOpInterface,
  SingleBlockImplicitTerminator<"plan::YieldOp">,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>
]> {
  let description = [{

    The `plan.inline_group` operation encapsulates a group of operations within
    its single-block region. The purpose of this operation is materialize
    in the IR the result of a clustering analysis.

    Each `plan.inline_group` operation must be assigned a `target` attribute
    that indicates how the cluster should be compiled / what "backend" will
    be targeted. Examples of "backends" could include entire sub-compiler
    stacks like TensorRT or an external library like CUBlas or CUDNN.

    The `plan.inline_group` operation is meant to be gradually lowered into a
    more specialized operation(s) and/or have the body outlined to a
    function-like operation such that the reuslts are replaced with a
    call-like operation.

  }];

  let arguments = (ins AnyAttr:$target);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
     `target` `(` $target `)` attr-dict-with-keyword (`->` type($results)^)? $body
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Attribute":$target,
      "llvm::function_ref<SmallVector<Value>(OpBuilder&, Location)>":$bodyBuilder
      ), [{
        $_state.addAttribute(getTargetAttrName($_state.name), target);
        Region *bodyRegion = $_state.addRegion();
        Block &body = bodyRegion->emplaceBlock();
        if(!bodyBuilder) {
          InlineGroupOp::ensureTerminator(*bodyRegion, $_builder, $_state.location);
        } else {
          OpBuilder::InsertionGuard g($_builder);
          $_builder.setInsertionPointToStart(&body);
          SmallVector<Value> toYield = bodyBuilder($_builder, $_state.location);
          $_builder.create<plan::YieldOp>($_state.location, toYield);
          $_state.addTypes(TypeRange(toYield));
        }
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// InlineClosedGroupOp
//===----------------------------------------------------------------------===//

def Plan_InlineClosedGroupOp : Plan_GroupOpBase<"inline_closed_group", [
  IsolatedFromAbove,
  AttrSizedOperandSegments,
  DestinationStyleOpInterface,
  SingleBlockImplicitTerminator<"plan::YieldOp">,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
    ["getEntrySuccessorOperands"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface,
    ["getAsmBlockArgumentNames"]>
]> {
  let description = [{

    The `plan.inline_closed_group` operation is similar to `dps.inline_group`
    except that it is isolated from above and explicitly captures destination
    operands. The `in_attrs` and `res_attrs` hold bounds attribute information
    for the input and result operands. The abcense of bounds information
    is allowed (`none` bounds).

    #### Example

    Consider the following simple program containing operations with
    dynamically shaped operands:

    ```mlir

    %0 = ... : tensor<?xf32> // A dynamically shaped operand
    %1 = .... : index        // A dynamic calculation of %0's extent

    %2 = plan.inline_group target(#plan.cluster_target<tensorrt>) {
      %2 = plan.with_shape %0 (%1) : (tensor<?xf32>, index) -> tensor<?xf32>
      %3 = stablehlo.exponential %2 : tensor<?xf32>
      yield %3 : tensor<?xf32>
    }

    ```

    It can be transformed into the below IR which explicitly captures all
    operands, operand shape bounds, and the allocation of a destination tensor
    based on the upper bound of the yielded values:


    ```mlir

    // Destination operand calculated from the upper bound of the
    // output.
    %empty = tensor.empty() : tensor<128xf32>

    //Slice %empty to actual linear size.
    %out = tensor.extract_slice %empty[0][%ub][1] :  tensor<128xf32> to tensor<?xf32>

    %2 = plan.inline_closed_group target(#plan.cluster_target<tensorrt>)
      inputs( %0, %1 : tensor<?xf32>, index)
      outs( %out : tensor<?xf32> )
      in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
      res_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]  -> tensor<?xf32> {
    ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
      %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
      %res = stablehlo.exponential %2 : tensor<?xf32>
      yield %res : tensor<?xf32>
    }

    ```

  }];
  let arguments = (ins Variadic<AnyTypeOf<[AnyRankedTensor, AnySignlessIntegerOrIndex]>>:$inputs,
                       Variadic<AnyRankedTensor>:$outs,
                       BoundsAttrArray:$input_attrs,
                       BoundsAttrArray:$res_attrs,
                       AnyAttr:$target);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `target` `(` $target `)` `\n`
    `inputs` `(` ( $inputs^ `:` type($inputs) `)` ) : ( `)` ) ?  `\n`
    `outs` `(` $outs `:` type($outs) `)` `\n`
    `in_attrs` $input_attrs `\n`
    `res_attrs` $res_attrs attr-dict-with-keyword `->` type($results)
     $body
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Attribute":$target,
                   "ValueRange":$inputs, "ValueRange":$outs,
                   CArg<"ArrayRef<BoundsAttr>", "{}">:$input_attrs,
                   CArg<"ArrayRef<BoundsAttr>", "{}">:$res_attrs)>
  ];

  let extraClassDeclaration = baseExtraClassDeclaration # [{

    MutableOperandRange getDpsInitsMutable() {
      return getOutsMutable();
    }

    /// Returns true if the `i-th` input argument has a tensor type.
    bool argHasTensorType(unsigned inputIdx) {
      assert(inputIdx < getInputs().size() && "input index out-of-bounds");
      return isa<RankedTensorType>(getInputs()[inputIdx].getType());
    }

    /// Returns the i-th input argument's bounds attribute.
    BoundsAttr getInputBoundsAttr(unsigned inputIdx) {
      assert(inputIdx < getInputs().size() && "input index out-of-bounds");
      return cast<BoundsAttr>(getInputAttrs()[inputIdx]);
    }

    ArrayRef<BlockArgument> getRegionOutArgs() {
      return getBody().getArguments().take_back(getOuts().size());
    }

    /// Populate the `res_attrs` from an array of BoundsAttrs.
    void setResAttrsAttr(ArrayRef<BoundsAttr> boundsAttrs) {
      setResAttrsAttr(::mlir::ArrayAttr::get(
        getOperation()->getContext(),
        ArrayRef<Attribute>(boundsAttrs.begin(), boundsAttrs.end())
      ));
    }

    /// Populate the `input_attrs` from an array of BoundsAttrs.
    void setInputAttrsAttr(ArrayRef<BoundsAttr> boundsAttrs) {
      setInputAttrsAttr(::mlir::ArrayAttr::get(
        getOperation()->getContext(),
        ArrayRef<Attribute>(boundsAttrs.begin(), boundsAttrs.end())
      ));
    }

  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Plan_YieldOp : Plan_Op<"yield", [
      Pure,
      Terminator,
      ReturnLike,
      ParentOneOf<["plan::InlineGroupOp",
                   "plan::InlineClosedGroupOp"]>]> {

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let builders = [
    OpBuilder<(ins)>
  ];
}

//===----------------------------------------------------------------------===//
// WithShapeOp
//===----------------------------------------------------------------------===//

def Plan_WithShapeOp : Plan_Op<"with_shape",
    [Pure,
     AllTypesMatch<["operand", "result"]>]> {
  let summary = "Ties a tensor value with index SSA values representing its shape";

  let description = [{

    The `plan.with_shape` operation is a utility operation that links a tensor
    SSA value with individual scalar Index-typed values representing the
    dimension sizes. The number of `shape` dimension values should be equal to
    the rank of `operand`.

    The operation returns a single result which is exactly equivalent to the
    operand. The purpose of the operation is therefore purely to link the SSA
    value to the dimension sizes.

    This operation is only really useful when `operand` has dynamic dimensions
    type, so if `operand` is statically shaped then this operation will be
    folded away.

  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let arguments = (ins AnyRankedTensor:$operand, Variadic<AnySignlessIntegerOrIndex>:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $operand `(` $shape `)` `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{

    /// Returns the SSA values that correspond only to the dynamic
    /// dimensions of `operand`.
    SmallVector<Value> getDynamicSizes() {
      SmallVector<Value> result;
      for(int64_t i = 0, e = getType().getRank(); i < e; i++) {
        if(getType().isDynamicDim(i))
          result.push_back(getShape()[i]);
      }
      return result;
    }

  }];
}

//===----------------------------------------------------------------------===//
// WithValuesOp
//===----------------------------------------------------------------------===//

def Plan_WithValuesOp : Plan_Op<"with_values",
    [Pure,
     DeclareOpInterfaceMethods<TensorKindOpInterface>,
     AllTypesMatch<["operand", "result"]>]> {
  let summary =
    "Ties a tensor value with index SSA values representing its element values";

  let description = [{
    The `plan.with_values` operation is a utility operation that links a tensor
    SSA value with individual scalar values representing the values of individual
    elements. The number of `elements` should be equal to the volume of the
    `operand`, which must have a static shape.

    The operation returns a single result which is exactly equivalent to the
    operand. The purpose of the operation is therefore purely to link the SSA
    value to the element values.
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let arguments = (ins StaticShapeTensorOf<[AnySignlessIntegerOrIndex]>:$operand,
                       Variadic<AnySignlessIntegerOrIndex>:$elements);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $operand `(` $elements `)` `:` custom<WithValuesTypes>(
      ref($elements), type($result), type($elements))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstraintOp
//===----------------------------------------------------------------------===//
def Plan_ConstraintOp : Plan_Op<"constraint", [Pure]>{
  let summary =
    "Used for applying constraints to its memref operand using attributes.";
  let description = [{
   This op is used to apply constraints to its memref operand using the 
   attributes associated for dimension analysis passes.
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$memrefToConstrain);

  let assemblyFormat = "`(` $memrefToConstrain `)` attr-dict  `:` type($memrefToConstrain)";
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANOPS_TD
