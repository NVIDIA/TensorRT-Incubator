#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTOPS_TD
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTOPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.td"

//===----------------------------------------------------------------------===//
// TensorRT Op Record Classes
//===----------------------------------------------------------------------===//

class TensorRT_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorRT_Dialect, mnemonic, !listconcat(traits, [TensorRTOpInterface])> {
  code trtLayerAdd = "";

  code baseClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
      return tensorrt::detail::isCompatibleReturnTypesShapes(lhs, rhs,
        /*shapesEqualUpToDynamicAmbiguity=*/true);
    }
  }];

  let extraClassDeclaration = baseClassDeclaration;
}

//===----------------------------------------------------------------------===//
// TensorRT Dialect Enums
//===----------------------------------------------------------------------===//
include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.td"

//===----------------------------------------------------------------------===//
// TensorRTModuleOp
//===----------------------------------------------------------------------===//

def TensorRT_TensorRTModuleOp : TensorRT_Op<"module", [
  IsolatedFromAbove, SymbolTable, Symbol] # GraphRegionNoTerminator.traits> {

  let summary =
    "A top level container for functions describing TensorRT programs";

  let description = [{
    A `tensorrt.module` is a container that encapsulates IR defining TensorRT
    programs. Each `func.func` in a TensorRT module should compile to a single
    TensorRT engine.

    The module has a symbol name so that operations outside the module
    can refer to the symbols names of nested functions.
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$name)>
  ];

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Return the name of this module if present.
    std::optional<StringRef> getName() { return getSymName(); }

    /// Construct a module from the location context.
    static TensorRTModuleOp create(Location loc, StringRef name);
  }];
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def TensorRT_CallOp : TensorRT_Op<"call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DestinationStyleOpInterface,
  AttrSizedOperandSegments,
  CallOpInterface
]> {
  let summary = "calls a TensorRT engine defined in a `tensorrt.module`";

  let arguments = (ins Variadic<AnyTypeOf<[AnyShaped, AnySignlessIntegerOrIndex]>>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       SymbolRefAttr:$callee);
  let results = (outs Variadic<AnyShaped>:$results);

  let extraClassDeclaration = [{
    /// Return the function representing the TRT engine that is being called.
    func::FuncOp getFuncCallee(SymbolTableCollection &symbolTable);

    // Declare the outputs as inits/outs to DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return getOperation()->getOperands();
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    MutableOperandRange getArgOperandsMutable() {
      return MutableOperandRange(getOperation());
    }
  }];

  let assemblyFormat = [{
    $callee `(` ($inputs^ `:` type($inputs))? `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`->` type($results)^)?
  }];
}

//===----------------------------------------------------------------------===//
// ActivationOp
//===----------------------------------------------------------------------===//
def TensorRT_ActivationOp : TensorRT_Op<"activation", [Pure,
      TensorRTInferTensorResultTypes, ElementwiseMappable]> {
  let summary = "TensorRT activation (IActivationLayer) operation";
  let description = [{
    The `tensorrt.activation` operation models a unary operation that depends
    on up to two parameters (`alpha` and `beta`). The purpose of the two
    parameters depends on the `activationType`.

    This op applies the `activationType` function on an element-wise basis
    to form the result.
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    TensorRT_ActivationTypeAttr:$activationType,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<F32Attr>:$beta
  );
  let results = (outs TensorRT_Tensor:$result);
  let assemblyFormat = "attr-dict  $input `:` type($input)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Return true if the `alpha` parameter should be provided
    /// for the current activation type.
    static bool requiresAlphaAttribute(ActivationType activationType);

    /// Return true if the `beta` parameter should be provided
    /// for the current activation type.
    static bool requiresBetaAttribute(ActivationType activationType);
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    nvinfer1::IActivationLayer *layer = $net->addActivation(
                                                    *$input,
                                                    *$activationType);
    if($alpha)
      layer->setAlpha(*$alpha);
    if($beta)
      layer->setBeta(*$beta);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def TensorRT_UnaryOp : TensorRT_Op<"unary", [Pure,
      TensorRTInferTensorResultTypes, ElementwiseMappable]> {
  let summary = "TensorRT unary (IUnaryLayer) operation";
  let description = [{
    The `tensorrt.unary` operation consists of applying single
    operation given by `unaryOperation` to the `input` elementwise.
    The result type is identical to the input type.
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    TensorRT_UnaryOperationAttr:$unaryOperation
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict $input `:` type($input)";

  let hasVerifier = 1;
  let hasFolder = 1;

  let trtLayerAdd = [{
    ::nvinfer1::IUnaryLayer *layer = $net->addUnary(*$input, *$unaryOperation);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
  }];
}


//===----------------------------------------------------------------------===//
// ElementWiseOp
//===----------------------------------------------------------------------===//

def TensorRT_ElementWiseOp : TensorRT_Op<"element_wise", [Pure,
      TensorRTInferTensorResultTypes,
      SameOperandsElementType,
      AllRanksMatch<["input1", "input2", "result"]>]>{
  let summary = "TensorRT element-wise (IElementWiseLayer) operation";
  let description = [{
    The `tensorrt.element_wise` operation takes two Tensor-typed inputs
    and performs a mathematical operation to yield the result in a per-element
    manner. This operation also has broadcast semantics described below.

    #### Broadcasting Semantics

    The input tensors must have the same rank. For each dimension, their lengths
    must match, or one of them must be one. In the latter case, the tensor is
    broadcast along that axis.

    The output tensor has the same rank as the inputs. For each output dimension,
    its length is equal to the lengths of the corresponding input dimensions if they match,
    otherwise it is equal to the length that is not one.

    Note that during compile time, the length of a dimension may be unknown.
    Therefore, it may not be possible to validate broadcasting semantics until
    runtime. For example:

    ```mlir
    tensorrt.element_wise kADD
        (%input0, %input1 : tensor<10x128xf32>, tensor<?x128xf32>) -> tensor<10x128xf32>
    ```

    In this case, the compiler will not reject the IR because it cannot prove
    that `?` is not `1`. But at runtime, TensorRT should throw an error if `?`
    does not turn out to be 1.
  }];

  let arguments = (ins
    TensorRT_Tensor:$input1,
    TensorRT_Tensor:$input2,
    TensorRT_ElementWiseOperationAttr:$elementwiseOperation
  );
  let results = (outs
    AnyRankedTensor:$result  );
  let assemblyFormat = [{
    attr-dict $elementwiseOperation
      `(` operands `:` type(operands) `)` `->` type(results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let trtLayerAdd = [{
    nvinfer1::IElementWiseLayer *layer = $net->addElementWise(
      *$input1, *$input2, *$elementwiseOperation);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def TensorRT_ConstantOp : TensorRT_Op<"constant", [Pure,
      AllTypesMatch<["weights", "result"]>,
      ConstantLike,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>{
  let summary = "TensorRT constant (IConstantLayer) operation";
  let description = [{
    The `tensorrt.constant` op represents a constant value.
  }];
  let arguments = (ins ElementsAttr:$weights);
  let results = (outs TensorRT_RankedTensorOf<[F16, F32, I32, BF16, TensorRT_I8,
    TensorRT_F8, TensorRT_I4]>:$result);
  let assemblyFormat = "attr-dict $weights";
  let hasVerifier = 1;
  let hasFolder = 1;

  let trtLayerAdd = [{
    nvinfer1::IConstantLayer *constLayer = $net->addConstant($resultShape, *$weights);
    nvinfer1::ITensor* result = constLayer->getOutput(0);
    if (!$e.isStronglyTyped()) {
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      constLayer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(result);
    $e.setMetadata(constLayer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//

def TensorRT_ConcatenationOp : TensorRT_Op<"concatenation",
      [Pure, SameOperandsAndResultElementType,
       TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT concatenate (IConcatenationLayer) operation";
  let description = [{
    The `tensorrt.concatenation` operation concatenates the given `inputs`
    along the dimension `axis`. The input types must have equal rank and
    the shapes must be such that dimension sizes other than the size for
    `axis` must be equal among all inputs.

    The concatenation axis is expected to be non-negative.

    The result dimension size along the concatenation axis is the sum of
    the corresponding input dimensions. Every other output dimension is
    the same as the corresponding dimension of the inputs.
  }];

  let arguments = (ins
    Variadic<TensorRT_RankedTensorOf<[Index, F32, F16, I32, BF16, TensorRT_I8, I1, TensorRT_F8]>>:$inputs,
    I32Attr:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;

  let trtLayerAdd = [{
    nvinfer1::IConcatenationLayer *layer =
        $net->addConcatenation($inputs.data(), $inputs.size());
    layer->setAxis(static_cast<int32_t>($axis));
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

def TensorRT_ConvolutionOp : TensorRT_Op<"convolution",
      [Pure, SameOperandsAndResultElementType, AttrSizedOperandSegments,
      InferTensorType]>{
  let summary = "TensorRT convolution (IConvolutionLayer) operation";
  let description = [{
    The `tensorrt.convolution` operation represents a 2D or 3D convolution:
    `input`'s spatial dimensions are convolved with the filters of `kernel`
    to form the result.

    The `input` is assumed to be organized such that the dimensions of the shape
    correspond to `[batch_dim, chan_dim, spatial dimensions... ]`.

    The `kernel` is assumed to be organized such that the dimensions of the shape
    correspond to `[num_filters, chan_dim / num_groups, filter spatial dimensions... ]`.

    The `kernel` input must be computable at build time via constant folding.

    An optional `bias` is supported, which adds a per-channel constant to each
    value in the output. If supplied `bias` must be the result of a
    `tensorrt.constant` operation.

    TODO: add other padding modes.
    TODO: add stride/padding/dilation/groups description.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16]>:$input,
    Optional<TensorRT_RankedTensorOf<[F32, F16, BF16]>>:$kernel,
    Optional<TensorRT_RankedTensorOf<[F32, F16, BF16]>>:$bias,
    OptionalAttr<ElementsAttr>:$kernelStatic,
    OptionalAttr<ElementsAttr>:$biasStatic,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pre_padding,
    DenseI64ArrayAttr:$post_padding,
    DefaultValuedAttr<UI32Attr, "1">:$num_groups,
    OptionalAttr<DenseI64ArrayAttr>:$dilation
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    attr-dict  `in` `(` $input `:` type($input) `)`
      (`kernel` `(` $kernel^ `:` type($kernel) `)` )?
      (`bias`   `(` $bias^   `:` type($bias)   `)` )?
      `->` type($result)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Return the number of spatial dimensions in the kernel.
    int64_t getNumSpatialDims() {
      return getType().getRank()-2;
    }

    ArrayRef<int64_t> getKernelSpatialShape() {
      return this->getKernelStatic().has_value()
            ? this->getKernelStatic()->getShapedType().getShape().take_back(
                  this->getNumSpatialDims())
            : this->getKernel().getType().getShape().take_back(
                  this->getNumSpatialDims());
    }
  }] # baseClassDeclaration;

  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$input, "OpFoldResult":$kernel, "OpFoldResult":$bias,
                   "ArrayRef<int64_t>":$stride,
                   "ArrayRef<int64_t>":$pre_padding,
                   "ArrayRef<int64_t>":$post_padding,
                   "uint32_t":$num_groups,
                   "std::optional<ArrayRef<int64_t>>":$dilation)>
  ];
  let trtLayerAdd = [{
    int32_t numOutputMaps = $op.getType().getDimSize(1);
    nvinfer1::IConvolutionLayer *layer = $net->addConvolutionNd(
        *$input, numOutputMaps,
        getNvInferDims($op.getKernelSpatialShape()),
        *$kernelStatic, *$biasStatic);
    if ($kernel)
      layer->setInput(1, *$kernel);
    if ($bias)
      layer->setInput(2, *$bias);
    layer->setStrideNd($stride);
    layer->setPrePadding($pre_padding);
    layer->setPostPadding($post_padding);
    if ($dilation)
      layer->setDilationNd(*$dilation);
    layer->setNbGroups($num_groups);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                            $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// EinsumOp
//===----------------------------------------------------------------------===//
def TensorRT_EinsumOp : TensorRT_Op<"einsum", [Pure,
  TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT einsum (IEinsumLayer) operation";
  let description = [{

    The `tensorrt.einsum` op implements a summation over the elements of the
    inputs along dimensions specified by the equation parameter, based on the
    Einstein summation convention.

    The layer can have one or two inputs of rank >= 0. All the inputs must have
    `f16` or `f32` data types, but can differ in data type.

    The result type has element type`f32` and a shape that is determined by
    the einsum equation.

    The equation specifies ASCII lower-case letters for each dimension in the
    inputs in the same order as the dimensions, separated by comma for each
    input. The dimensions labeled with the same subscript must match or be
    broadcastable. Repeated subscript labels in one input take the diagonal.
    Repeating a label across multiple inputs means that those axes will be
    multiplied. Omitting a label from the output means values along those axes
    will be reduced (summed).

    There are two modes of interpretation of  the equation: implicit and
    explicit.

    #### Implicit Mode

    In implicit mode, the indices which appear once in the expression will be
    part of the output in increasing
    alphabetical order.

    #### Explicit Mode

    In explicit mode, the output can be controlled by specifying output
    subscript labels by adding an arrow (`->`) followed by subscripts for
    the output. For example, “ij,jk->ik” is equivalent to “ij,jk”.

    #### Examples

    Many common operations can be expressed using the Einsum equation.
    For example:
    Matrix Transpose:             ij->ji
    Sum:                          ij->
    Matrix-Matrix Multiplication: ik,kj->ij
    Dot Product:                  i,i->
    Matrix-Vector Multiplication: ik,k->i
    Batch Matrix Multiplication:  ijk,ikl->ijl

    #### Unsupported einsum features

    The following einsum conventions are not supported by TensorRT
    (as of TensorRT 8.5):
    - Diagonal (repeated index in one operand)
    - Use of ellipsis
  }];

  let arguments = (ins
    Variadic<TensorRT_RankedTensorOf<[F32, F16, BF16]>>:$inputs,
    StrAttr:$equation
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type($result)";

  let trtLayerAdd = [{
    nvinfer1::IEinsumLayer *layer = network->addEinsum(
        $inputs.data(), $inputs.size(), $equation.str().c_str());
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def TensorRT_GatherOp : TensorRT_Op<"gather",
      [Pure, AllElementTypesMatch<["data", "result"]>, TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT gather (IGatherLayer with kDEFAULT mode) operation";
  let description = [{
    The `tensorrt.gather` operation represents a gather operation on the input
    data elements `data` using the indices `indices`. The operation additionally
    has two parameters:

    1. `numBroadcastDims`: controls how the dimensions of `indices` are
        interpreted. It must be either 0 (default) or 1 and follow `numBroadcastDims` <= `axis`.
    2. `axis`: The dimension number in the shape of `data`'s type from which
        to start gathering data.

    The rank of the `indices` tensor has no restrictions.

    The result is expected to be equal to
    `rank(data) + rank(indices) - 1 - numBroadcastDims`.

    First `numBroadcastDims` dimensions of the output are computed by
    applying broadcast rules to first `numBroadcastDims` dimensions of
    the indices indices and data tensors. Note that `0 <= numBroadcastDims <= 1`.
    The rest of dimensions are computed by copying the dimensions of the `data`
    and replacing the `axis` dimension with the dimensions of the corresponding
    indices.

    This op follows the semantics of
    [tf.GatherV2](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/gather-v2)
    and [onnx.Gather](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)
    and [torch.index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html)
    (when the rank of `indices` is constrained to be `1`).

    #### Output Computation when `numBroadcastDims = 0`:

    For data of rank `D`, indices of rank `I`, the result rank `R := D + I -1`.

    Let `J = (r_0, ..., r_{axis-1}, r_{axis},..., r_{axis+I-1}, ..., r_{R-1})` be a point
    in the iteration space of `result`.

    For each `J`, the slice of these numbers `J[axis, axis+I-1] = (r_{axis}, ..., r_{axis+I-1})`
    is an iteration point in the space of `indices`.

    Then let `K = index[r_{axis}, ..., r_{axis+I-1} ]` and the result is formed by

    ```
    result[r_0, ...,  r_{R-1} ]
        = data[r_0, ..., r_{axis-1}, K, r_{axis+I}, ..., r_{R-1} ]
    ```

    #### Caveats

    * If an dimension of `data` has dynamic/unknown length, then using a negative
      index for it has undefined behavior.
    * This layer does not support DLA.
    * Zero will be stored for all out-of-bounds (OOB) access.
  }];

   let arguments = (ins
    TensorRT_Tensor:$data,
    TensorRT_Tensor:$indices,
    I64Attr:$axis,
    DefaultValuedAttr<I64Attr, "0">:$numBroadcastDims
  );
  let results = (outs TensorRT_Tensor:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` operands `:` type(operands) `)` `->` type($result)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the string representation of the TensorRT
    /// "GatherMode" enum value.
    static StringRef getGatherModeString() {
      return "kDEFAULT";
    }

    /// Return the (TensorRT MLIR Dialect) enum value for the gather mode.
    /// Note that this is different from the `nvinfer1::GatherMode` enum, which
    /// depends on a particular TensorRT version.
    static ::mlir::tensorrt::GatherMode getGatherMode() {
      return ::mlir::tensorrt::GatherMode::kDEFAULT;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    nvinfer1::IGatherLayer *layer = $net->addGatherV2(
        *$data, *$indices, nvinfer1::GatherMode::kDEFAULT);
    layer->setGatherAxis(static_cast<int32_t>($axis));
    layer->setNbElementWiseDims(static_cast<int32_t>($numBroadcastDims));
    layer->setMode(nvinfer1::GatherMode::kDEFAULT);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// GatherNdOp
//===----------------------------------------------------------------------===//

def TensorRT_GatherNdOp : TensorRT_Op<"gather_nd",
      [Pure, AllElementTypesMatch<["data", "result"]>, TensorRTInferTensorResultTypes]>{

  let summary = "TensorRT gather-ND operation (IGatherLayer with kND mode)";

  let description = [{

    The `tensorrt.gather_nd` operation gathers slices of `data` into the
    `result`. The operation can be described using the below pseudo-code.
    The key difference between `tensorrt.gather` and `tensorrt.gather_nd` is that
    the indices of `tensorrt.gather` specify an scalar offset into a particular
    axis while the indices of `tensorrt.gather` specify an N-dimensional index.
    `N` is less than or equal to the rank of `data`:

    ```
    let r = rank(data)
    let q = rank(indices)
    let index_vector_size = shape(indices)[-1]

    let result_shape = concat( shape(indices)[:-1], shape(data)[-index_vector_size:] )
    let index_volume = volume(shape(indices)[:-1])

    let result = tensor_of(shape=result_shape, element_type=element_type(data))

    for i in 0..index_volume:
      let index_coord = *delin(i, shape(indices)[:-1])
      // Slice 1D 'offsets' from indices.
      let offsets = indices[*index_coord]
      assert rank(offsets) == 1
      // Use `offsets` to select a (r-index_vector_size) slice
      // from `data`.
      slice_offsets = concat(offsets, [0]*(r-index_vector_size))
      slice_sizes = concat([1]*len(offsets), shape(data)[-index_vector_size:])
      slice_strides = [1] * r
      data_slice = slice(data, slice_offsets, slice_sizes, slice_strides)
      result[concat(index_coord, [0] * (r-index_vector_size)] = data_slice
    ```

    #### Constraints:

      - `index_vector_size == shape(indices)[-1] <= rank(data)`
      - `rank(result) == q + r - 1 - shape(indices)[-1]`
      - `shape(result) == concat(shape(indices)[0:-1], shape(data)[-index_vector_size:])`
      - `element_type(result) == element_type(data)`

    #### Example:


    ```mlir
    %0 = tensorrt.gather_nd data(%arg0) indices(%arg1)
      : (tensor<1024x32x4xf32>, tensor<16x17x2xi32>) -> tensor<16x17x4xf32>
    ```

  }];

  let arguments = (ins
    TensorRT_Tensor:$data,
    TensorRT_Tensor:$indices
  );

  let results = (outs TensorRT_Tensor:$result);

  let assemblyFormat = [{
    attr-dict `data` `(` $data `)`  `indices` `(` $indices `)`
     `:` functional-type(operands, results)
  }];

  let trtLayerAdd = [{
    nvinfer1::IGatherLayer *layer = $net->addGatherV2(
        *$data, *$indices, nvinfer1::GatherMode::kND);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// GatherElementsOp
//===----------------------------------------------------------------------===//

def TensorRT_GatherElementsOp : TensorRT_Op<"gather_elements",
      [Pure, AllRanksMatch<["data", "indices", "result"]>, TensorRTInferTensorResultTypes]> {
  let summary = "TensorRT gather (IGatherLayer with kELEMENTS mode) operation";
  let description = [{
    The `tensorrt.gather_elements` operation represents a gather operation on
    the input data elements `data` using the indices `indices`. The operation
    additionally has a parameter`axis`, which is the dimension number in
    the shape of `data` from which to start gathering data.

    This op follows the semantics of
    [onnx.GatherElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements).

    The shape of the result is equal to the shape of `indices`.

    For each element X of indices:
      Let J denote a sequence for the subscripts of X
      Let K = sequence J with element [axis] replaced by X
        output[J] = data[K]

    #### Output Calculation

    For each point `(i_0, ..., i_{rank-1})` in the iteration space of `indices`:

    Let `idx = indices[i_0, ..., i_{rank-1} ]`.

    Then the result values are calculated as follows:

    ```
    result[i_0, ..., i_{axis}, ..., i_{rank-1} ] = data[i_0, ..., idx, ..., i_rank{-1} ]
    ```

    #### Example

    Given a tensor `input : tensor<2x20xf32>`, suppose you wanted to permute
    the elements of `input[0, :]` and `input[1, :]` according to two
    separate permutations `perm1 : tensor<20xi64>` and `perm2 : tensor<20xi64>`.

    Form the `indices : tensor<2x20xi64>` such that  `indices[0, :] = perm1`
    and `indices[1, :] = perm2`.

    Then the desired result can be achieved via

    ```mlir
    %result = tensorrt.gather_elements {
        axis = 1 : i64
      } ins(%input, %indices : tensor<2x20xf32>, tensor<2x20xi64>)
        -> tensor<2x20xf32>
    ```

  }];

  let arguments = (ins
    TensorRT_Tensor:$data,
    TensorRT_Tensor:$indices,
    I64Attr:$axis
  );
  let results = (outs TensorRT_Tensor:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` operands `:` type(operands) `)` `->` type($result)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the string representation of the TensorRT
    /// "GatherMode" enum value.
    static StringRef getGatherModeString() {
      return "kELEMENTS";
    }

    /// Return the (TensorRT MLIR Dialect) enum value for the gather mode.
    /// Note that this is different from the `nvinfer1::GatherMode` enum, which
    /// depends on a particular TensorRT version.
    static ::mlir::tensorrt::GatherMode getGatherMode() {
      return ::mlir::tensorrt::GatherMode::kELEMENT;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    nvinfer1::IGatherLayer *layer = $net->addGatherV2(
        *$data, *$indices, nvinfer1::GatherMode::kELEMENT);
    layer->setMode(nvinfer1::GatherMode::kELEMENT);
    layer->setGatherAxis(static_cast<int32_t>($axis));
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// Identity84Op
//===----------------------------------------------------------------------===//

def TensorRT_Identity84Op : TensorRT_Op<"identity84", [Pure,
      TensorRTPartiallyInferTensorResultTypes,
      ElementwiseMappable]>{
  let summary = "TensorRT identity (IIdentityLayer) operation (TensorRT 8.4 version)";
  let description = [{
    The `tensorrt.identity` operation performs a cast from the the input tensor
    type to the result tensor type, changing only the element type.

    This operation corresponds to the TensorRT 8.4 specification.

    Valid conversions for TensorRT 8.4 are:

      (f32 | f16 | i32 | bool) -> (f32 | f16 | i32)
  }];

  let arguments = (ins TensorRT_RankedTensorOf<[F32, F16, I32, I1]>:$input);
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, I32]>:$result);
  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";
  let hasFolder = 1;

  let trtLayerAdd = [{
    nvinfer1::IIdentityLayer *layer = $net->addIdentity(*$input);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def TensorRT_IdentityOp : TensorRT_Op<"identity", [
      Pure,
      TensorRTPartiallyInferTensorResultTypes,
      ElementwiseMappable]>{
  let summary = "TensorRT identity (IIdentityLayer) operation";
  let description = [{
    The `tensorrt.identity` operation performs a cast from the the input tensor
    type to the result tensor type, changing only the element type.

    This operation corresponds to the TensorRT 8.5 specification.

    Valid conversions are as follows:

        (f32 | f16 | bf16 | i32 | bool) -> (f32 | f16 | bf16 | i32 | bool)
        (f32 | f16) -> (uint8)
        (uint8) -> (f32 | f16)
  }];

  let arguments = (ins TensorRT_RankedTensorOf<[Index, F32, F16, I32, BF16, TensorRT_I8, UI8, I1]>:$input);
  let results = (outs TensorRT_RankedTensorOf<[Index, F32, F16, I32, BF16, TensorRT_I8, UI8, I1]>:$result);
  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;

  let trtLayerAdd = [{
    FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(outputTrtType))
      return failure();
    if (!$e.isStronglyTyped()) {
      nvinfer1::IIdentityLayer *layer = $net->addIdentity(*$input);
      layer->setOutputType(
          0, *outputTrtType);
      $results.push_back(layer->getOutput(0));
      $e.setMetadata(layer, $op);
    } else {
      nvinfer1::ICastLayer *layer = $net->addCast(*$input,
          *outputTrtType);
      $results.push_back(layer->getOutput(0));
      $e.setMetadata(layer, $op);
    }
  }];
}

//===----------------------------------------------------------------------===//
// LinspaceOp
//===----------------------------------------------------------------------===//

def TensorRT_LinspaceOp : TensorRT_Op<"linspace", [
      Pure,
      AttrSizedOperandSegments,
      TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT linspace (a variant of FillLayer) operation";
  let description = [{
    Generate an output tensor `result` using the "linspace" operation,
    described below.

    Let the output rank be `n` and let `s` be a tensor that is a
    slice of the result along dimension `i`. Then the contents of
    `s` are given by:

    `s[j] = start + step[i] * j` if `step` is a tensor or
    `s[j] = start + step * j` if `step` is a scalar.

    The inputs `start` and `step` must either both be given by
    tensor-typed SSA values or both be given by static attributes. One
    cannot be static and the other dynamic.

    The shape can either be static and given by the shape of the
    result type, or it can be given by the `shape` input. If the
    result has unknown dimensions, then the `shape` parameter
    must be present.
  }];

  let arguments = (ins
    Optional<1DTensorOf<[I32]>>:$shape,
    Optional<0DTensorOf<[F32, I32]>>:$start,
    Optional<1DTensorOf<[F32, I32]>>:$step,
    OptionalAttr<F64Attr>:$static_start,
    OptionalAttr<F64Attr>:$static_step
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, I32]>:$result);
  let assemblyFormat = [{
    attr-dict
    `[`  ($start^ `:` type($start)) : ($static_start)? `]`
    `[`  ($shape^ `:` type($shape)) : (`static`)? `]`
    `[`  ($step^  `:` type($step)) : ($static_step)? `]`
    `:` type($result)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the TensorRT enum value describing the IFillLayer variant type.
    FillOperation getFillOperationEnumValue() {
      return tensorrt::FillOperation::kLINSPACE;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(outputTrtType))
      return failure();
    nvinfer1::ILayer *layer =
      $e.addFillLayer(
        *outputTrtType,
        $resultShape, $shape, nvinfer1::FillOperation::kLINSPACE,
        $static_start, $static_step,
        $start, $step);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

def TensorRT_RandomUniformOp : TensorRT_Op<"random_uniform", [
      Pure,
      AttrSizedOperandSegments,
      TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT random uniform (a variant of FillLayer) operation";
  let description = [{
    Generates an output tensor with samples drawn from a uniform distribution.
    Range ([low, high)) of the distribution is give by `low` and `high` inputs.
    Both of these inputs are either tensor-typed SSA values or static attributes.
    If nothing is provided, default value of `low` is 0 and `high` is 1.

    The shape can either be static and given by the shape of the
    result type, or it can be given by the `shape` input. If the
    result has unknown dimensions, then the `shape` parameter
    must be present.
  }];

  let arguments = (ins
    Optional<1DTensorOf<[I32]>>:$shape,
    Optional<0DTensorOf<[F32, F16]>>:$low,
    Optional<0DTensorOf<[F32, F16]>>:$high,
    OptionalAttr<F64Attr>:$static_low,
    OptionalAttr<F64Attr>:$static_high
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16]>:$result);
  let assemblyFormat = [{
    attr-dict
      (`low` `(` $low^ `:` type($low) `)` )?
      (`high` `(` $high^   `:` type($high)  `)` )?
      (`shape` `(` $shape^ `:` type($shape) `)` )?
      `->` type($result)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the TensorRT enum value describing the IFillLayer variant type.
    FillOperation getFillOperationEnumValue() {
      return tensorrt::FillOperation::kRANDOM_UNIFORM;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(outputTrtType))
      return failure();
    nvinfer1::ILayer *layer =
      $e.addFillLayer(
        *outputTrtType,
        $resultShape, $shape, nvinfer1::FillOperation::kRANDOM_UNIFORM,
        $static_low, $static_high, $low, $high);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

def TensorRT_RandomNormalOp : TensorRT_Op<"random_normal", [
      Pure,
      AttrSizedOperandSegments,
      TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT random normal (a variant of FillLayer) operation";
  let description = [{
    Generates an output tensor with samples drawn from a normal (Gaussian)
    distribution. The mean of the normal distribution is give by `mean`
    and the standard deviation is given by `std` inputs.

    Both of these inputs are either tensor-typed SSA values or static
    attributes.If nothing is provided, default value of `mean` is 0 and
    `std` is 1 i.e. samples are drawn from the standard normal distribution,
    by default.

    The shape can either be static and given by the shape of the
    result type, or it can be given by the `shape` input. If the
    result has unknown dimensions, then the `shape` parameter
    must be present.
  }];

  let arguments = (ins
    Optional<1DTensorOf<[I32]>>:$shape,
    Optional<0DTensorOf<[F32, F16]>>:$mean,
    Optional<0DTensorOf<[F32, F16]>>:$std,
    OptionalAttr<F64Attr>:$static_mean,
    OptionalAttr<F64Attr>:$static_std
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16]>:$result);
  let assemblyFormat = [{
    attr-dict
      (`mean` `(` $mean^ `:` type($mean) `)` )?
      (`std` `(` $std^   `:` type($std)  `)` )?
      (`shape` `(` $shape^ `:` type($shape) `)` )?
      `->` type($result)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the TensorRT enum value describing the IFillLayer variant type.
    FillOperation getFillOperationEnumValue() {
      return tensorrt::FillOperation::kRANDOM_NORMAL;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(outputTrtType))
      return failure();
    nvinfer1::ILayer *layer =
      $e.addFillLayer(
        *outputTrtType,
        $resultShape, $shape, nvinfer1::FillOperation::kRANDOM_NORMAL,
        $static_mean, $static_std, $mean, $std);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// NormalizationOp
//===----------------------------------------------------------------------===//

def TensorRT_NormalizationOp : TensorRT_Op<"normalization",
        [Pure, TensorRTInferTensorResultTypes,
        AllRanksMatch<["input", "scale", "bias", "result"]>,
        AllElementTypesMatch<["input", "scale", "bias", "result"]>]>{
  let summary = "TensorRT normalization (INormalizationLayer) operation";
  let description = [{
    The `tensorrt.normalization` layer normalizes an input tensor using the
    following equation.
    ```
    result = ((input-E(input, a))/(sqrt(V(input, a))+eps)) * scale + bias
    ```
    where,
    - `input` = input tensor.
    - E(input, a) = mean of the input tensor along `a` axis.
    - sqrt = square root operation.
    - V(input, a) = variance of the input tensor along `a` axis.
    - `eps` = a small value added to the denominator for numerical stability.
    Default is `1e-5`.
    - `scale` = scale tensor.
    - `bias` = bias tensor.
    - `axis` (a) = axis to perform normalization on.

    Value of the attribute `num_groups` (default is 1) decides the type of
    normalization. If `num_groups != 1`, the input channels will be split into
    `num_groups` before normalization is performed. The channel dimension is
    considered to be the second dimension.

    In general, the following holds true for different types of normalizations.
    1. Batch Normalization
        - Input: [N, C, H, W, ..]
        - Axis: 0
        - Scale and Bias: [1, C, 1, 1, ..., 1]
        - Mean and standard-deviation are computed over the batch dimensions.
    2. Group Normalization (num_groups = G)
        - Input: [N, C, H, W, ...]
        - Grouped Input: [N, G, C//G, H, W, ...]
        - Axis: [2, 3, 4, ....]
        - Scale and Bias: [1, G, 1, 1, ... , 1]
        - Input channels are split into `G` groups, each with `C/G` channels. Mean
        and standard-deviation are computed for each group separately.
        - If G=1, this is equivalent to the Layer Normalization.
        - If G=C, this is equivalent to the Instance Normalization.
    3. Instance Normalization
        - Input: [N, C, H, W, ..]
        - Axis: [2, 3, 4, ....]
        - Scale and Bias: [1, C, 1, 1, ..., 1]
        - Mean and standard-deviation are computed per dimension for each object in
        a batch.
    4. Layer Normalization
        - Input: [D0, D1, D2, D3 .... DN]
        - Axis: [2, 3, .. N]
        - Scale and Bias: [1, 1, D2, D3, D4 ... DN]
        - Mean and standard-deviation are computed over the last D dimensions.
        - In general, if axis selected forms `normalized` shape ([Di, Di+1, ...],
        starting at index `i`), scale and bias has [1, 1, .. 1i, normalized] shape.
        Scale/bias shape has dim 1 up-to `i`th dim from which normalization axis
        starts.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F16, F32, BF16]>:$input,
    TensorRT_RankedTensorOf<[F16, F32, BF16]>:$scale,
    TensorRT_RankedTensorOf<[F16, F32, BF16]>:$bias,
    TensorRT_DimensionListAttr:$axis,
    OptionalAttr<F32Attr>:$eps,
    OptionalAttr<I32Attr>:$num_groups
  );
  let results = (outs TensorRT_RankedTensorOf<[F16, F32, BF16]>:$result);
  let assemblyFormat =[{
    attr-dict `(`$input `:` type($input) `,` $scale `:` type($scale) `,` $bias `:` type($bias)`)` `->` type($result)
  }];
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::INormalizationLayer* layer = $net->addNormalization(
      /*input=*/*$input,
      /*scale=*/*$scale,
      /*bias=*/*$bias,
      /*axisMask=*/$axis);
    if (eps)
      layer->setEpsilon(*$eps);
    if (num_groups)
      layer->setNbGroups(*$num_groups);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

def TensorRT_PoolingOp : TensorRT_Op<"pooling",
      [Pure, SameOperandsAndResultElementType, InferTensorType]>{
  let summary = "TensorRT pooling (IPoolingLayer) operation";
  let description = [{
    The `tensorrt.pooling` operation applies a windowed reduction over the
    input. By convention, the input tensor type's shape dimensions are
    assumed to be in the form `[batch_dim, channel_dim, spatial_dims...]`.

    The mathematical operation for the windowed reduction is given by the
    `poolingType` attribute. See the `PoolingType` enum definition for an
    enumeration of the various supported operations.

    The reduction window size is given by `windowSize` and should be of
    size 2 or size 3.

    TODO: Currently only explicit pre/post padding is supported, but the
    underlying TensorRT layer also supports automatic padding styles

    The `blendFactor` attribute is only relevant when `poolingType` is
    set to `kMAX_AVERAGE_BLEND`.

    The `averageCountExcludesPassing` is only relevant when `poolingType`
    is set to `kMAX_AVERAGE_BLEND` or `kAVERAGE`. When set, elements of
    the input that correspond to elements added during padding are not
    counted toward the total when performing an average calculation for
    a given window position.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, TensorRT_I8]>:$input,
    DenseI64ArrayAttr:$windowSize,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$prePadding,
    DenseI64ArrayAttr:$postPadding,
    TensorRT_PoolingTypeAttr:$poolingType,
    OptionalAttr<F32Attr>:$blendFactor,
    OptionalAttr<BoolAttr>:$averageCountExcludesPadding
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, TensorRT_I8]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` $input `:` type($input) `)` `->` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Return the number of spatial dimensions of the input/result.
    int64_t getNumSpatialDims() {
      return getType().getRank()-2;
    }
  }] # baseClassDeclaration;

  let trtLayerAdd = [{
    nvinfer1::IPoolingLayer* layer = $net->addPoolingNd(
      /*input=*/*$input,
      /*type=*/*$poolingType,
      /*windowSize=*/$windowSize);
    layer->setPrePadding($prePadding);
    layer->setPostPadding($postPadding);
    layer->setStrideNd($stride);
    if(averageCountExcludesPadding) {
      layer->setAverageCountExcludesPadding(*$averageCountExcludesPadding);
    }
    if(blendFactor) {
      assert(*$poolingType == nvinfer1::PoolingType::kMAX_AVERAGE_BLEND);
      layer->setBlendFactor(*$blendFactor);
    }
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

def TensorRT_ReduceOp : TensorRT_Op<"reduce",
      [Pure, SameOperandsAndResultElementType, TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT reduction operation";
  let description = [{
    The `tensorrt.reduce` operation models the TensorRT IReduceLayer semantics.

    The operation reduces one or more axes of the `input` tensor as described
    by the attribute `reduceAxes`, which is an array of dimension indices.
    NOTE: This differs from the underlying TensorRT API, which requires the
    reduction axes to be specified as a bitmask. The conversion to a bitmask
    occurs automatically during translation to the C++ or TensorRT target.

    The `keepDimensions` attribute specifies whether the result should be
    rank-reduced (`keepDimensions = false`) or the same rank as the input
    (`keepDimensions = true`). When not rank-reduced, the result shape
    dimensions that were reduced by the operation are equal to `1`.

    The `reduceOperation` attribute specifies what reduction operation is
    performed (see the `ReduceOperation` enum definition).
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16, I32]>:$input,
    TensorRT_DimensionListAttr:$reduceAxes,
    DefaultValuedAttr<BoolAttr, "false">:$keepDimensions,
    TensorRT_ReduceOperationAttr:$reduceOperation
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16, I32]>:$result);
  let assemblyFormat = [{
    $reduceOperation $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;

  let hasCanonicalizer = 1;

  let trtLayerAdd = [{
    nvinfer1::IReduceLayer *layer = $net->addReduce(
        *$input, *$reduceOperation, $reduceAxes, $keepDimensions);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def TensorRT_SelectOp : TensorRT_Op<"select", [
      Pure,
      TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT ISelectLayer operation";
  let description = [{
    The `tensorrt.select` operation accepts a boolean
    tensor and produces an output that is elementwise equal to
    `thenInput` where `condition` is true and otherwise equal to
    `elseInput`.

    The input shapes may be broadcasted according to TensorRT's
    broadcasting semantics. The element types of `thenInput` and
    `elseInput` must be equal and also equal to the element type of
    the result.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[I1]>:$condition,
    TensorRT_RankedTensorOf<[I32, F16, F32, BF16]>:$thenInput,
    TensorRT_RankedTensorOf<[I32, F16, F32, BF16]>:$elseInput
  );
  let results = (outs TensorRT_RankedTensorOf<[I32, F16, F32, BF16]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type($result)";
  let hasVerifier = 1;

  let trtLayerAdd = [{
    ::nvinfer1::ISelectLayer *layer =
        $net->addSelect(*$condition, *$thenInput, *$elseInput);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

def TensorRT_SliceOp : TensorRT_Op<"slice", [
      Pure,
      AttrSizedOperandSegments,
      TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT slice (ISliceLayer) operation";
  let description = [{
    The `tensorrt.slice` operation slices an input tensor based on
    offset, size, and stride values.

    The offset, size, and stride values can be either a static list of
    i32 values, or each of them can come from a rank-1 i32 tensor-typed
    SSA value.

    The slice layer selects for each dimension a start location from
    within the input tensor specified by `start`/`static_start`, and
    copies elements to the output tensor using the specified `stride`/
    `static_stride` across the input tensor.

    The `mode` attribute specifies how the slice behaves at out-of-bounds
    (OOB) access. When `mode` is kCLAMP or kREFLECT, for each input
    dimension, if its size is 0 then the corresponding output
    dimension must be 0 too.
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    Optional<TensorRT_Tensor>:$fill,
    Optional<TensorRT_ShapeTensor>:$start,
    Optional<TensorRT_ShapeTensor>:$size,
    Optional<TensorRT_ShapeTensor>:$stride,
    OptionalAttr<DenseI32ArrayAttr>:$static_start,
    OptionalAttr<DenseI32ArrayAttr>:$static_size,
    OptionalAttr<DenseI32ArrayAttr>:$static_stride,
    DefaultValuedAttr<TensorRT_SliceModeAttr, "tensorrt::SliceMode::kDEFAULT">:$mode
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input
     `[` ( $start^ `` `:` type($start) ) : ( `` custom<StaticIndexI32Array>($static_start) ) ? `]` ``
     `[` ( $size^ `` `:` type($size) ) : ( `` custom<StaticIndexI32Array>($static_size) ) ? `]` ``
     `[` ( $stride^ `` `:` type($stride) ) : ( `` custom<StaticIndexI32Array>($static_stride) ) ? `]`
     ( `fill` `(` $fill^ `:` type($fill) `)` )?
    attr-dict `:` type($input) `to` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Required for AttrSizedOperandSegments interface: return the number of leading
    /// operands before the variadic/optional operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Returns true if all slice parameters are static.
    bool hastStaticSliceParameters() {
      return this->getStaticStart().has_value() &&
             this->getStaticSize().has_value() &&
             this->getStaticStride().has_value();
    }

    /// Return true if the slice start (offset) is static.
    bool hasStaticSliceStart() {
      return this->getStaticStart().has_value();
    }

    /// Return true if the slice stride is static.
    bool hasStaticSliceSize() {
      return this->getStaticSize().has_value();
    }

    /// Return true if the slice stride is static.
    bool hasStaticSliceStride() {
      return this->getStaticStride().has_value();
    }

    /// Return the dynamic or static start as OpFoldResult.
    OpFoldResult getStartAsOpFoldResult() {
      if(!hasStaticSliceStart())
        return getStart();
      return getStaticStartAttr();
    }

    /// Return the dynamic or static size as OpFoldResult.
    OpFoldResult getSizeAsOpFoldResult() {
      if(!hasStaticSliceSize())
        return getSize();
      return getStaticSizeAttr();
    }

    /// Return the dynamic or static stride as OpFoldResult.
    OpFoldResult getStrideAsOpFoldResult() {
      if(!hasStaticSliceStride())
        return getStride();
      return getStaticStrideAttr();
    }
  }] # baseClassDeclaration;

  let builders = [
    // Builder for static slice parameters.
    OpBuilder<(ins "Value":$input, "ArrayRef<int32_t>":$start,
                   "ArrayRef<int32_t>":$size, "ArrayRef<int32_t>":$stride,
                   CArg<"SliceMode", "SliceMode::kDEFAULT">:$sliceMode,
                   CArg<"Value", "Value()">:$fill)>,
    // Builder using static array for start/stride and Value for size.
    OpBuilder<(ins "Value":$input, "ArrayRef<int32_t>":$start,
                   "Value":$size, "ArrayRef<int32_t>":$stride,
                   CArg<"SliceMode", "SliceMode::kDEFAULT">:$sliceMode,
                   CArg<"Value", "Value()">:$fill)>,
    // Builder for static or dynamic cases using OpFoldResult -- inferred type
    OpBuilder<(ins "Value":$input, "OpFoldResult":$start,
                   "OpFoldResult":$size, "OpFoldResult":$stride,
                   CArg<"SliceMode", "SliceMode::kDEFAULT">:$sliceMode,
                   CArg<"Value", "Value()">:$fill)>,
    // Builder for static or dynamic cases using OpFoldResult -- specified type
    OpBuilder<(ins "Type":$result, "Value":$input, "OpFoldResult":$start,
                   "OpFoldResult":$size, "OpFoldResult":$stride,
                   CArg<"SliceMode", "SliceMode::kDEFAULT">:$sliceMode,
                   CArg<"Value", "Value()">:$fill)>,
  ];

  let trtLayerAdd = [{
    ::nvinfer1::ISliceLayer *layer =
      $net->addSlice(*$input, kNullDims, kNullDims, kNullDims);
    layer->setMode(*$mode);
    if (!$static_start) {
      layer->setInput(1, *$start);
    } else {
      layer->setStart(*$static_start);
    }
    if (!$static_size) {
      layer->setInput(2, *$size);
    } else {
      layer->setSize(*$static_size);
    }
    if (!$static_stride) {
      layer->setInput(3, *$stride);
    } else {
      layer->setStride(*$static_stride);
    }
    if ($fill) {
      layer->setInput(4, *$fill);
    }
    $results.push_back(layer->getOutput(0));
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// SoftMaxOp
//===----------------------------------------------------------------------===//

def TensorRT_SoftMaxOp : TensorRT_Op<"softmax", [Pure,
      TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT softmax (ISoftMaxLayer) operation";
  let description = [{

    The `tensorrt.softmax` operation applies the "softmax" to the
    `axis` dimension of the input tensor. The shape of the result tensor is
    equal to the shape of the input tensor. The computation that yields the
    result tensor is described below.

    Let the input tensor be of type `tensor<d_1 x ... x d_axis x ... x d_n x f32>`.
    Let `x := input_{i_1, ..., i_axis, ..., i_n}` be a point in the input
    tensor. The point `y` in the result with the same coordinates is given by

    ```
    y = e^{x} \ ( sum_{i_axis=0}^{d_axis - 1} input_{i_1, ..., i_axis, ..., i_n} )
    ```

    In other words, the result is the pointwise exponential of the input, normalized
    along the `axis` dimension.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16]>:$input,
    I64Attr:$axis
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16]>:$result);
  let assemblyFormat = "attr-dict $input `:` type($input)";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    ::nvinfer1::ISoftMaxLayer *layer = $net->addSoftMax(*$input);
    // `setAxes` expects a bitmask.
    layer->setAxes(1U << $axis);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// OneHotOp
//===----------------------------------------------------------------------===//

def TensorRT_OneHotOp : TensorRT_Op<"one_hot", [
    Pure, TensorRTInferTensorResultTypes, AllElementTypesMatch<["values", "result"]>]>{
  let summary = "TensorRT IOneHotLayer operation";
  let description = [{

    The `tensorrt.one_hot` operation requires 'indices', 'values', 'depth', and 'axis' as as inputs.
    The locations represented by the index values in 'indices' will have 'on_value' in the output tensor and
    the other locations will have 'off_value'.
    Here, 'on_value' and 'off_value' are specified by 'values' of format ['off_value', 'on_value'].
    The rank of the output tensor will be one greater than the rank of the input tensor.
    The additional dimension in the output tensor is of size 'depth'.
    As mentioned before, only one of 'depth' elements in the dimension is 'on_value' and the others are 'off_value'.
    The additional dimension will be inserted at the position specified by 'axis'.

    Example 1:
    When indices = [0, 2, 4], values = [0, 1], depth = 5, and axis = -1 (the innermost dimension)
    the output is
      [[1  0  0  0  0]
       [0  0  1  0  0]
       [0  0  0  0  1]].

    Example 2:
    When indices = [[0, 1], [2, 3], [4, 1]], values = [0, 2], depth = 5, and axis = -1 (the innermost dimension)
    the output is
      [[[2  0  0  0  0], [0  2  0  0  0]],
       [[0  0  2  0  0], [0  0  0  2  0]],
       [[0  0  0  0  2], [0  2  0  0  0]]].

    Example 3:
    When indices = [0, 2, 4], values = [0, 2], depth = 5, and axis = 0
    the output is
      [[2  0  0]
       [0  0  0]
       [0  2  0]
       [0  0  0]
       [0  0  2]].
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[I32]>:$indices,
    TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8, I1, I32]>:$values,
    TensorRT_RankedTensorOf<[I32]>:$depth,
    SI64Attr:$axis
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8, I1, I32]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    ::nvinfer1::IOneHotLayer *layer = $net->addOneHot(*$indices, *$values, *$depth, $axis);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// RaggedSoftMaxOp
//===----------------------------------------------------------------------===//

def TensorRT_RaggedSoftMaxOp : TensorRT_Op<"ragged_softmax", [
  Pure, TensorRTInferTensorResultTypes, AllElementTypesMatch<["input", "result"]>]>{
  let summary = "TensorRT IRaggedSoftMaxLayer operation";
  let description = [{

    The `tensorrt.ragged_softmax` operation requires `input` and `bounds` as input tensors
    and produces `output` as the output tensor. The `input` is of ZxS and
    the `bounds` is Zx1, which specifies the length of each of the Z sequences.
    The shape of `output` is the same as `input`.
    This layer computes a softmax across each of the Z sequences for the specified length.
    Values outside of the specified lengths are set to zero.

    As of TRT 8.6.10.0, `input` and `bounds` must be a 3D tensor in the explicit batch mode,
    or a 2D tensor in the implicit batch mode, which is deprecated but exists for backward compatibility.

    Example:
    When `input` is of 1x3x5
     [[[1.0, 1.0,  1.0,  1.0, 1.0],
       [1.0, 1.0,  1.0,  1.0, 0.5],
       [3.0, 4.0, -2.0, 10.0, 5.0]]]
    and `bounds` is of 1x3x1
     [[[5], [4], [3]]],

    the `output` is of 1x3x5
     [[[0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
       [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
       [0.2685, 0.7297, 0.0018, 0.0000, 0.0000]]]
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, BF16, TensorRT_I8]>:$input,
    TensorRT_RankedTensorOf<[I32]>:$bounds
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, BF16, TensorRT_I8]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    ::nvinfer1::IRaggedSoftMaxLayer *layer = $net->addRaggedSoftMax(*$input, *$bounds);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// AssertionOp
//===----------------------------------------------------------------------===//

def TensorRT_AssertionOp : TensorRT_Op<"assertion", [Pure]>{
  let summary = "TensorRT IAssertionLayer operation";
  let description = [{
    An assertion layer in a network

    The `tensorrt.assertion` operation has an input tensor, `condition` and no output.
    The input tensor must be a boolean type of rank 0 or 1.
    If any element of the input is provably false at build time, the network is rejected.
    If any element of the input is false at runtime for the supplied runtime dimensions,
    an error occurs, much the same as if any other runtime error is handled.

    The attribute named `message` defines the message to print if the assertion fails.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[I1]>:$condition,
    StrAttr:$message
  );
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)`";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    ::nvinfer1::IAssertionLayer *layer = $net->addAssertion(*$condition, $message.str().c_str());
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// MatrixMultiplyOp
//===----------------------------------------------------------------------===//

def TensorRT_MatrixMultiplyOp : TensorRT_Op<"matrix_multiply", [Pure,
  SameOperandsAndResultElementType, TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT IMatrixMultiplyLayer operation";
  let description = [{

    The `tensorrt.matrix_multiply`layer represents a Matrix Multiplication C = A * B.

    When A and B are matrices or vectors, computes the inner product A * B:

        matrix * matrix -> matrix
        matrix * vector -> vector
        vector * matrix -> vector
        vector * vector -> scalar

    Matrix Operation (tensorrt::MatrixOperation) attribute defines the operations performed in a tensor before multiplication.
    - kNONE
    Treat x as a matrix if it has two dimensions, or as a collection of matrices if x has more than two dimensions,
    where the last two dimensions are the matrix dimensions. x must have at least two dimensions.
    - kTRANSPOSE
    Like kNONE, but transpose the matrix dimensions.
    - kVECTOR
    Treat x as a vector if it has one dimension, or as a collection of vectors if x has more than one dimension. x
    must have at least one dimension.
    The first input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
    with dimensions [M, 1, K] with MatrixOperation::kNONE.
    The second input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
    with dimensions [M, K, 1] with MatrixOperation::kNONE.

    Matrix/Vector dimensions can be thought as combination of collection dimensions and inner product dimensions.
    - For Matrices, all but the last two dimensions are collection dimensions.
    - For Vectors, all but the last dimension are collection dimensions.
    - The number of collection dimensions MUST be equal.
    - Broadcast rules* apply to the collection dimensions.

    * For each collection dimension, the lengths must match, or one of them must be equal to 1. In the latter case, the tensor
    is broadcast along that axis.

    The output rank is the number of collection dimensions plus 0..2 dimensions for the inner products. 0 for vector*vector,
    1 for vector*matrix or converse and 2 for matrix*matrix.

    Inputs MUST have same datatype but can have unequal rank if one of them is kVECTOR.

    Example,
    ```mlir
    func.func @trt_matrix_vector(%arg0: tensor<1x64xf32>, %arg1: tensor<10x64x128xf32>)
            -> tensor<10x128xf32> {
      %0 = tensorrt.matrix_multiply {
        op0 = #tensorrt.matrix_operation<kVECTOR>,
        op1 = #tensorrt.matrix_operation<kNONE>
      } ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<10x64x128xf32>) -> tensor<10x128xf32>
      return %0 : tensor<10x128xf32>
    }
    ```
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8, I1]>:$input0,
    TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8, I1]>:$input1,
    TensorRT_MatrixOperationAttr:$op0,
    TensorRT_MatrixOperationAttr:$op1
  );
  let results = (outs
    AnyRankedTensor:$result  );
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    /// Get the number of collection dims.
    static ArrayRef<int64_t> getCollectionDimsImpl(TensorType input,
                                           tensorrt::MatrixOperation op) {
      if (op == tensorrt::MatrixOperation::kVECTOR)
        return input.getShape().drop_back(1);
      return input.getShape().drop_back(2);
    };

    ArrayRef<int64_t> getCollectionDims(int64_t inpIdx = 0) {
      TensorType input = inpIdx ? getInput1().getType() : getInput0().getType();
      tensorrt::MatrixOperation op = inpIdx ? getOp1() : getOp0();
      return getCollectionDimsImpl(input, op);
    };

    /// Get collection rank
    int64_t getCollectionRank(int64_t inpIdx = 0) {
      TensorType input = inpIdx ? getInput1().getType() : getInput0().getType();
      tensorrt::MatrixOperation op = inpIdx ? getOp1() : getOp0();
      if (op == tensorrt::MatrixOperation::kVECTOR)
        return input.getShape().size() - 1;
      return input.getShape().size() - 2;
    };

  }] # baseClassDeclaration;
  let trtLayerAdd = [{
    ::nvinfer1::IMatrixMultiplyLayer *layer = $net->addMatrixMultiply(
      *$input0, *$op0, *$input1, *$op1);
    $e.setMetadata(layer, $op);
    if (!$e.isStronglyTyped()) {
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
      ::nvinfer1::IIdentityLayer *identityLayer = $net->addIdentity(*layer->getOutput(0));
      identityLayer->setOutputType(0, *outputTrtType);
      $e.setMetadata(identityLayer, $op);
      $results.push_back(identityLayer->getOutput(0));
    } else {
      $results.push_back(layer->getOutput(0));
    }
  }];
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

def TensorRT_TopKOp : TensorRT_Op<"top_k",
      [Pure,
       TensorRTInferTensorResultTypes,
       AllRanksMatch<["input", "values", "indices"]>,
       AllShapesMatch<["values", "indices"]>]> {
  let summary = "TensorRT top K (ITopKLayer) operation";
  let description = [{

    The `tensorrt.top_k` operation finds the top K largest or smallest elements
    and their indices from each slice  of the tensor along the `axis`
    dimension. The two results correspond to the values and indices,
    respectively.

    The `topkOperation` is either `kMAX` or `kMIN` and determines whether the
    K largest or K smallest values are returned.

    The result values and indices are sorted with respect to the values:

    `kMAX`: sorted from largest to smallest value
    `kMAX`: sorted from smallest to largest value

    The result of the operation are never rank-reduced, even if `K=1`.

    #### Examples:

    ```mlir

    %0 = tensorrt.top_k {
      topkOperation = #tensorrt.topKOperation<kMAX>,
      k = 1 : i64,
      axis = 2 : i64
    } %input  : tensor<100x200xf32> -> (tensor<100x1xf32>, <100x1xi32>)

    ```
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    I64Attr:$k,
    I64Attr:$axis,
    TensorRT_TopKOperationAttr:$topkOperation
  );
  let results = (outs
    AnyRankedTensor:$values, TensorRT_RankedTensorOf<[I32]>:$indices);
  let assemblyFormat = "$topkOperation attr-dict  $input `:` type($input) `->` type(results)";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$k, "int64_t":$axis,
                   "tensorrt::TopKOperation":$topkOperation)>
  ];

  let trtLayerAdd = [{
    ::nvinfer1::ITopKLayer *layer = $net->addTopK(
        *$input, *$topkOperation,
        static_cast<int32_t>($k),
        static_cast<uint32_t>(1U << $axis));
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> valuesTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getValues().getType().getElementType());
      if (failed(valuesTrtType))
        return failure();
      layer->setOutputType(0, *valuesTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $results.push_back(layer->getOutput(1));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// PaddingOp
//===----------------------------------------------------------------------===//

def TensorRT_PaddingOp : TensorRT_Op<"padding", [Pure, TensorRTInferTensorResultTypes]> {
  let summary = "TensorRT padding(IPaddingLayer) operation";
  let description = [{

    The `tensorrt.padding` operation adds zero-padding at the start and end of
    the input tensor and store the result in an output tensor.

    When `prePadding` and `postPadding` are positive, the tensor is pad with
    zeros, otherwise it is trimmed. The amount of padding/trimming to use at
    the start and end of each dimension is defined by `prePadding` and
    `postPadding` respectively. It only supports padding/trimming along exactly
    two innermost dimensions.

    Each output dimension `O_{i}` is calculated based on input dimension `I_{i}`
    prePadding and postPadding as follows,
    n = rank(input)
    O_{i} = I_{i}                                  for 0 <= i < (n - 2)
    O_{i} = I_{i} + prePadding[j] + postPadding[j] for (n - 2) <= i < n and j = i - ( n- 2)

    #### Examples:

    ```mlir

    %0 = tensorrt.padding {
      prePadding = array<i64: 3, -2>,
      postPadding = array<i64: 2, -4>
    } %input  : tensor<1x1x10x10xf32> -> tensor<1x1x15x4xf32>

    ```

    #### Caveat
    Padding exactly two innermost dimensions is supported.

  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, I32, TensorRT_I8]>:$input,
    DenseI64ArrayAttr:$prePadding,
    DenseI64ArrayAttr:$postPadding
  );
  let results = (outs TensorRT_RankedTensorOf<[F16, F32, I32, TensorRT_I8]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";
  let hasVerifier = 1;

  let trtLayerAdd = [{
    ::nvinfer1::IPaddingLayer *layer = $net->addPaddingNd(
        *$input, $prePadding, $postPadding);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// NonZeroOp
//===----------------------------------------------------------------------===//

def TensorRT_NonZeroOp : TensorRT_Op<"non_zero",
      [Pure, TensorRTInferTensorResultTypes]> {
  let summary = "TensorRT NonZero (INonZeroLayer) operation";
  let description = [{

    `tensorrt.non_zero` layer computes the indices of the input tensor where the value is non zero.
    The `result` has shape [input rank, #of non-zero elements] and contains indices of non-zero elements
    in row major order.

    This operation follows same semantics as that of `onnx.NonZero` and is similar to `tf.Where`.

    #### Examples:

    ```mlir
    %0 = tensorrt.non_zero %input  : tensor<100x200xf32> -> tensor<2x?xi32>
    ```
  }];

  let arguments = (ins TensorRT_Tensor:$input);
  let results = (outs TensorRT_RankedTensorOf<[I32]>:$result);
  let assemblyFormat = "attr-dict  $input `:` type($input) `->` type($result)";

  let trtLayerAdd = [{
    nvinfer1::INonZeroLayer *layer = $net->addNonZero(*$input);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def TensorRT_IfOp : TensorRT_Op<"if", [Pure, TensorRTInferTensorResultTypes,
      DeclareOpInterfaceMethods<RegionBranchOpInterface, [
                                  "getEntrySuccessorRegions",
                                  "getRegionInvocationBounds"]>,
      NoRegionArguments]>{
  let summary = "TensorRT if (IIfConditionalLayer) operation";
  let description = [{
    The `tensorrt.if` operation represents an "if-then-else" type control flow
    operation. The condition determines which of the "true branch" or "false
    branch" executes. Only one of the branches execute at runtime (i.e. the
    branches are not "eagerly" executed).

    The result types yielded from both branches must be equivalent and
    equal to the result type of the `tensorrt.if` operation.

    This operation obeys the following convention: if an operation is within the
    region, then it is lazily executed. By using any SSA value defined above,
    that SSA value becomes a "conditional branch input" in the
    [terminology of TensorRT API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-conditionals).

    In other words, unlike the TensorRT API, there is not a
    "ConditionalInputLayer" abstraction because this is captured implicitly through
    the nested structure of the IR. If the intent is to execute an operation
    "eagerly" regardless of the conditional value, then it should be
    moved out of the conditional body.

    #### Example

    ```mlir
    func.func @trt_if(%arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
      %cond = arith.constant dense<1> : tensor<i1>
      %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
          %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
              -> tensor<10xf32>
          tensorrt.yield %arg1: tensor<10xf32>
        } else {
          %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
              -> tensor<10xf32>
          tensorrt.yield %arg1: tensor<10xf32>
        }
      return %result: tensor<10xf32>
    }
    ```
  }];

  let arguments = (ins 0DTensorOf<[I1]>:$condition);

  let results = (outs Variadic<TensorRT_Tensor>:$results);

  let regions = (region SizedRegion<1>:$trueRegion, AnyRegion:$falseRegion);

  let assemblyFormat = [{
    `(` $condition `:` type($condition) `)` attr-dict `->` type($results)
    $trueRegion
    `else`
    $falseRegion
  }];

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$condition,
      "function_ref<void(OpBuilder &, Location)>":$trueBranchBuilder,
      "function_ref<void(OpBuilder &, Location)>":$falseBranchBuilder
    )>
  ];

  let trtLayerAdd = [{
    ::nvinfer1::IIfConditional* layer = $net->addIfConditional();
    layer->setCondition(*$condition);
    {
      NvInferNetworkEncoder::TensorMapScope ifScope($e.getTensorMap());

      {
        SetVector<Value> definedAbove;
        mlir::getUsedValuesDefinedAbove($op->getRegions(), definedAbove);
        for (Value use : definedAbove) {
          assert($e.contains(use) != 0 && "expected use to be remapped");
          nvinfer1::IIfConditionalInputLayer *inputLayer = layer->addInput(
              *$e.lookup(use));
          if (!$e.isStronglyTyped()){
            FailureOr<nvinfer1::DataType> useTrtType = getNvInferDataType($op.getLoc(), use.getType());
            if (failed(useTrtType))
              return failure();
            inputLayer->setOutputType(0, *useTrtType);
          }
          $e.map(use, inputLayer->getOutput(0));
        }
      }

      // Traverse true branch.
      if(failed($e.encodeRegion($trueRegion)))
        return $op->emitOpError("failed to encode true region");
      // Traverse false branch.
      if (failed($e.encodeRegion($falseRegion)))
        return $op->emitOpError("failed to encode false region");

      // Add the output
      {
        Operation *trueTerminator = $op.getTrueRegion().front().getTerminator();
        Operation *falseTerminator = $op.getFalseRegion().front().getTerminator();
        assert($op->getNumResults() == trueTerminator->getNumOperands() &&
              falseTerminator->getNumOperands() == $op.getNumResults());
        for (auto [opResult, trueVal, falseVal] :
            llvm::zip($op.getResults(), trueTerminator->getOperands(),
                      falseTerminator->getOperands())) {
          assert($e.contains(trueVal) > 0 && $e.contains(falseVal) > 0);
          ::nvinfer1::IIfConditionalOutputLayer *outputLayer =
              layer->addOutput(*$e.lookup(trueVal), *$e.lookup(falseVal));
          if (!$e.isStronglyTyped()){
            FailureOr<nvinfer1::DataType> resultTrtType = getNvInferDataType($op.getLoc(),
                                                                opResult.getType());
            if (failed(resultTrtType))
              return failure();
            outputLayer->setOutputType(0, *resultTrtType);
          }
          $results.push_back(outputLayer->getOutput(0));
        }
      }
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//
def TensorRT_ConditionOp : TensorRT_Op<"condition",
    [Pure, Terminator, HasParent<"tensorrt::WhileOp">,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
      ["getSuccessorRegions"]>]> {

  let summary = "While loop's continuation condition. ";

  let description = [{
    The `tensorrt.condition` operation accepts the continuation of
    `tensorrt.while` construct. If its first argument is true, the
    "body" region of the `tensorrt.while` is executed, with the remaining
    arguments forwarded to the entry block of the region. Otherwise,
    the loop terminates.
  }];

  let arguments = (ins 0DTensorOf<[I1]>:$condition, Variadic<AnyType>:$args);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = [{
    `(` $condition `:` type($condition) `)` attr-dict ($args^ `:` type($args))?
  }];
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
def TensorRT_WhileOp : TensorRT_Op<"while", [Pure,
      DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands"]>]> {

  let summary = "TensorRT While Loop (ILoopLayer) operation";
  let description = [{
    The `tensorrt.while` operation represents a while-style loop that executes its
    body while a condition is satisfied.

    The `tensorrt.while` operation consists of two regions: "condition" region and
    "body" region.

    #### Example

    The below example constructs a simple while loop that performs an operation while
    the given input condition is true and saves the result to an output tensor.
    Here, the input condition checks that iter variable is less than limit and
    when the condition is satisfied, the sum of iter and one tensors is executed.

    ```mlir
    func.func @trt_while_loop() -> (tensor<f32>) {
      %one = tensorrt.constant dense<1.0> : tensor<f32>
      %iter_init = tensorrt.constant dense<1.0> : tensor<f32>
      %limit = tensorrt.constant dense<10.0> : tensor<f32>
      %result0 = tensorrt.while {} (%iter_init : tensor<f32>) -> tensor<f32>
      {
        // condition
        ^bb0(%iter:tensor<f32>):
          %cond = tensorrt.element_wise <kLESS>(%iter, %limit : tensor<f32>, tensor<f32>)
                -> tensor<i1>
          tensorrt.condition(%cond : tensor<i1>) %iter : tensor<f32>
      } ,
      {
        // body
        ^bb1(%iter:tensor<f32>):
          %new_iter = tensorrt.element_wise <kSUM> (%one, %iter : tensor<f32>, tensor<f32>) -> tensor<f32>
          tensorrt.yield %new_iter: tensor<f32>
      }
      return %result0 : tensor<f32>
    }
    ```
  }];

   let arguments = (ins Variadic<TensorRT_Tensor>:$operands);

   let regions = (region SizedRegion<1>:$condRegion, SizedRegion<1>:$bodyRegion);

   let results = (outs Variadic<TensorRT_Tensor>:$results);

   let hasVerifier = 0;

   let assemblyFormat = [{
     attr-dict  `(` $operands `:` type($operands) `)` `->` type($results)
      $condRegion `,`
      $bodyRegion
    }];

   let hasRegionVerifier = 1;

   let trtLayerAdd = [{
    using namespace nvinfer1;
    ILoop *loop = $net->addLoop();
    NvInferNetworkEncoder::TensorMapScope whileBodyScope($e.getTensorMap());
    SetVector<Value> definedAbove;
    mlir::getUsedValuesDefinedAbove($op->getRegions(), definedAbove);

    // Add recurrenceLayers for loop carried dependencies for both body and
    // condition regions.
    SmallVector<ILayer *> recurrenceLayers;
    SmallVector<ILayer *> identityLayers;

    addRecurrenceLayersForWhileLoopCarries($net, loop, $operands,
                                           recurrenceLayers, identityLayers);
    // Process condition Region
    $e.map($op.getWhileConditionLoopCarriedDeps(), identityLayers);
    if (failed($e.encodeRegion($condRegion)))
      return $op->emitOpError("failed to encode while op's condition region");

    // Define loop stopping condition
    Value condition = $op.getBlockTerminatorInCondRegion()->getOperand(0);
    ITensor *tripLimit = $e.lookup(condition);
    loop->addTripLimit(*tripLimit, ::nvinfer1::TripLimit::kWHILE);

    // Process Body Region after mapping its loop carries.
    $e.map($op.getWhileBodyLoopCarriedDeps(), identityLayers);
    if (failed($e.encodeRegion($bodyRegion)))
      return $op->emitOpError("failed to encode while op's body region");

    // Map output layer to results
    Operation *term = $op.getBlockTerminatorInBodyRegion();
    for (auto [termOperand, recurrenceLayer] :
         llvm::zip(term->getOperands(), recurrenceLayers)) {
      recurrenceLayer->setInput(1, *$e.lookup(termOperand));
      ILoopOutputLayer *outLayer = loop->addLoopOutput(
          *recurrenceLayer->getOutput(0), ::nvinfer1::LoopOutput::kLAST_VALUE);
      $results.push_back(outLayer->getOutput(0));
    }
   }];

  let extraClassDeclaration = [{
    MutableArrayRef<BlockArgument> getWhileConditionLoopCarriedDeps() {
      return getCondRegion().getArguments();
    }
    MutableArrayRef<BlockArgument> getWhileBodyLoopCarriedDeps() {
      return getBodyRegion().getArguments();
    }
    mlir::Operation *getBlockTerminatorInCondRegion() {
      return getCondRegion().getBlocks().front().getTerminator();
    }
    mlir::Operation *getBlockTerminatorInBodyRegion() {
      return getBodyRegion().getBlocks().front().getTerminator();
    }
   }];

}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//
def TensorRT_ForOp : TensorRT_Op<"for", [Pure,
      DeclareOpInterfaceMethods<LoopLikeOpInterface, [
        "getLoopInductionVars", "getLoopLowerBounds", "getLoopSteps",
        "getLoopUpperBounds"]>,
      DeclareOpInterfaceMethods<RegionBranchOpInterface, [
        "getEntrySuccessorOperands"
      ]>,
      SingleBlockImplicitTerminator<"tensorrt::YieldOp">]>{
  let summary = "TensorRT Loop (ILoopLayer) operation";
  let description = [{
    The `tensorrt.for` operation represents a for-style loop that executes its
    body once for each valid value of the induction variable.

    The iteration variable takes values in [`start`, `stop`), incrementing by
    `step` each iteration. The values `start`, `stop`, and `step` must be
    `tensor<i32>` (0-D i32 tensor types).

    The loop body is a single-block region. The block arguments represent
    the following (in order from first to last):

     1. The induction variable.
     2. Loop carried dependencies.

    #### Loop carried dependencies

    The initial values of all loop carried dependencies are given by the op's
    `init` arguments (see example). For each loop carried dependency, the loop
    body must yield a value, representing the value that should be passed to
    the next iteration.

    #### Example

    The below example constructs a simple loop that adds a tensor of 1's
    to the input argument 10 times:

    ```mlir
    func.func @trt_for_loop(%arg0: tensor<10xf32>) -> tensor<10xf32> {
      %lb = tensorrt.constant dense<0> : tensor<i32>
      %ub = tensorrt.constant dense<10> : tensor<i32>
      %step = tensorrt.constant dense<1> : tensor<i32>
      %ones = tensorrt.constant dense<1.0> : tensor<10xf32>
      %0 = tensorrt.for %i = %lb to %ub step %step init(%iter0 = %arg0) -> tensor<10xf32> {
        %add = tensorrt.element_wise <kSUM>(%iter0, %ones : tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
        tensorrt.yield %add : tensor<10xf32>
      }
      return %0 : tensor<10xf32>
    }
    ```

  }];

  let arguments = (ins 0DTensorOf<[I32]>:$lb,
                       0DTensorOf<[I32]>:$ub,
                       0DTensorOf<[I32]>:$step,
                       Variadic<TensorRT_Tensor>:$init);
  let results = (outs Variadic<TensorRT_Tensor>:$results);

  let regions = (region SizedRegion<1>:$body_region);

  let hasCustomAssemblyFormat = 1;

  let hasRegionVerifier = 1;

  let trtLayerAdd = [{
    using namespace nvinfer1;
    NvInferNetworkEncoder::TensorMapScope forBodyScope($e.getTensorMap());
    ILoop* loop = $net->addLoop();
    ITensor* ivBlockArg = addForStyleInductionVariable($net, loop,
      $lb, $ub, $step);
    $e.map($op.getInductionVar(), ivBlockArg);

    // Add other block args. We setup the recurrence input for these
    // at the yield translation.
    SmallVector<ILayer *> recurrenceLayers = addRecurrenceLayersForBlockArgs(
                                                                  loop, $init);
    $e.map($op.getBlockArgsForLoopCarriedDeps(), recurrenceLayers);

    // Encode the loop body.
    if(failed($e.encodeRegion($body_region)))
      return $op->emitOpError("failed to encode for op body region");

    // Set the second input for the recurrence block args.
    Operation *term = $op.getBody()->getTerminator();
    for(auto [termOperand, recurrenceLayer] : llvm::zip(term->getOperands(),
                                                   recurrenceLayers)) {
      recurrenceLayer->setInput(1, *$e.lookup(termOperand));
      ILoopOutputLayer *outLayer = loop->addLoopOutput(
          *recurrenceLayer->getOutput(0), ::nvinfer1::LoopOutput::kLAST_VALUE);
      $results.push_back(outLayer->getOutput(0));
    }
  }];

  let extraClassDeclaration = [{
    BlockArgument getInductionVar() {
      return getBody()->getArgument(0);
    }

    MutableArrayRef<BlockArgument> getBlockArgsForLoopCarriedDeps() {
      return getBody()->getArguments().drop_front(1);
    }
  }];

  let extraClassDefinition = [{
    //===----------------------------------------------------------------------===//
    // LoopLikeOpInterface Methods
    //===----------------------------------------------------------------------===//
    SmallVector<Region*> $cppClass::getLoopRegions() {
      return {&getBodyRegion()};
    }
    std::optional<SmallVector<OpFoldResult>> $cppClass::getLoopLowerBounds() {
      return SmallVector<OpFoldResult>{getLb()};
    }
    std::optional<SmallVector<OpFoldResult>> $cppClass::getLoopUpperBounds() {
      return SmallVector<OpFoldResult>{getUb()};
    }
    std::optional<SmallVector<OpFoldResult>> $cppClass::getLoopSteps() {
      return SmallVector<OpFoldResult>{getStep()};
    }
    std::optional<SmallVector<Value>> $cppClass::getLoopInductionVars() {
      return SmallVector<Value>{getInductionVar()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

def TensorRT_ShapeOp : TensorRT_Op<"shape", [Pure,
  TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT shape operation";
  let description = [{
    The `tensorrt.shape` operation returns the shape of a given `input` tensor
    as a `i32` tensor.

    The `result` tensor is a rank-1 i32 tensor. The shape of the result tensor
    is `tensor<[inputRank]xi32>`. Note that for scalar tensors (e.g. of type
    `tensor<f32>`), the result type is `tensor<0xi32>`.

    #### Example

    ```mlir

    %1 = tensorrt.shape %arg0 : tensor<10x?x?xf32> -> tensor<3xi32>
    %2 = tensorrt.shape %arg1 : tensor<f32> -> tensor<0xi32>

    ```
  }];

  let arguments = (ins TensorRT_RankedTensorOf<[Index, F32, F16, BF16, I32, TensorRT_I8, I1, TensorRT_F8]>:$input);
  let results = (outs 1DTensorOf<[I32]>:$result);

  let assemblyFormat = "attr-dict $input `:` type($input) `->` type($result)";

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let trtLayerAdd = [{
    ::nvinfer1::IShapeLayer* layer = $net->addShape(*$input);
    $e.setMetadata(layer, $op);
    $results.push_back(layer->getOutput(0));

  #if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    // Cast i64 to i32, since that is what we currently support.
    // TODO: update shape semantics to have `index` type, which is determined
    // during translation by target TRT version.
    ::nvinfer1::ICastLayer *cast = $net->addCast(*layer->getOutput(0),
      nvinfer1::DataType::kINT32);
    $e.setMetadata(cast, $op);
    $results.back() = cast->getOutput(0);
  #endif
  }];
}

//===----------------------------------------------------------------------===//
// ParametricReLUOp
//===----------------------------------------------------------------------===//

def TensorRT_ParametricReLUOp : TensorRT_Op<"parametric_relu", [Pure,
    AllRanksMatch<["input", "slope"]>, TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT ParametricReLU(IParametricReLULayer) operation";
  let description = [{
    The `tensorrt.parametricrelu` applies the function `f` to each element `x` at indices`i=(i_0, ... i_n)` of the input tensor,
    where `f` is given as 'f(x) := x if x > 0, otherwise `slope[i_0,...,i_n]` * x`

    #### Example

    ```mlir

    %result = tensorrt.parametric_relu ins(%input, %slope : tensor<2x20xf32>, tensor<2x20xf32>)
        -> tensor<2x20xf32>

    ```
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8]>:$input,
    TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8]>:$slope
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16, TensorRT_I8]>:$result);
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";
  let hasVerifier = 1;

  let trtLayerAdd = [{
    ::nvinfer1::IParametricReLULayer  *layer = $net->addParametricReLU(
        *$input, *$slope);
    $results.push_back(layer->getOutput(0));
  }];
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//
def TensorRT_ShuffleOp : TensorRT_Op<"shuffle",
    [Pure, TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT IShuffleLayer operation";
  let description = [{
    The `tensorrt.shuffle` layer applies the following operations in sequence:

      1. Transpose the input (`first_transpose`)
      2. Reshape the result of (1) - the specified shape may be either
         static (`reshape`) or dynamic (`dynamic_reshape`).
      3. Transpose the result of (2).

    This operation tries to follow exactly the semantics of the underlying
    TensorRT "IShuffleLayer" operation. When applying a pure reshape or
    transpose, it is recommended to use one of the extension operations
    "tensorrt.reshape" or "tensorrt.transpose".

    #### Caveats

    #### Scalar reshapes

    It is legal to shape to a scalar (e.g. `tensor<1xf32>` to `tensor<f32>`)
    and vice-versa.

    #### `-1` in the Reshape Specification

    The value `-1` in the reshape specification infers that dimension
    by inspecting the source shape and the rest of the reshape dimensions.

    If 'zero_is_placeholder' is false, it is invalid for the specified shape
    to contain both a 0 and a -1, since the value of the dimension
    corresponding to -1 cannot be determined uniquely.

    #### "Zero is Placeholder" Mode

    **WARNING**: the "zero is placeholder" mode follows the underlying TRT
    mode, which can have surprising and unintuitive behavior. The intent
    of this mode historically was allow direct correspondence to frameworks
    that also has this mode, e.g. Cafe. Although it is not deprecated as of
    TRT 8.5, it is not recommended to use this mode unless absolutely necessary.

    When `zero_is_placeholder = true`, 0-values in the reshape specification
    are interpreted as being placeholders for input dimensions.

    In this mode, an inference procedure is followed to identify which input
    dimension corresponds to each "0". The procedure is described as follows:

    1. Let `i` be the index of some "0" in the `reshape` array.
    2. Let `f(i)` be the corresponding dimension in the source shape (the
       transposed input).
    3. Let `f` be the identity: `f(i) = i`.

    Any case where this results in out-of-bounds access for **any zero**
    in the reshape specification is considered invalid.

    Any case where this results in a shape whose volume does not equal
    the volume of the source shape is considered invalid.

    To illustrate this point, consider the following examples, where
    "I" is the source shape, "R" is the reshape specification, and "O"
    is the shape after reshaping. When the result is invalid (and will
    result in verification failure)

      1. I: [1, 2, 3, 4]
         R: [0, -1, 0]
         O: [1, 8, 3]

      2. I: [1, 2, 3, 4]
         R: [1, 0, 0, 0]
         O: [1, 2, 3, 4]

      3. I: [1, 2, 3, 4]
         R: [0, 2, 0, 0, 1]
         O: [1, 2, 3, 4, 1]

      4. I: [1, 2, 3, 4]
         R: [0, 1, 0, 0]
         O: [1, 1, 3, 4] INVALID (VOLUME)

      5. I: [30, 20, 10, 1]
         R: [600, 0, 0, 0, 0]
         O: [1, 2, 3, 4, _] INVALID (OOB)

      6. I: [30, 20, 10, 1]
         R: [600, 0, 0]
         O: [600, 20, 10] INVALID (VOLUME)

    #### Examples:

    ```mlir
    func.func @trt_shuffle(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32> {
      %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2, 3>,
        reshape = array<i64: 2, 3, 224, 224>,
        second_transpose = array<i64: 0, 1, 2, 3>
      } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>
      return %0 : tensor<2x3x224x224xf32>
    }

    func.func @trt_shuffle_infer(%arg0: tensor<3x4x6xf32>) -> tensor<3x24xf32> {
      %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 3, -1>,
        second_transpose = array<i64: 0, 1>
      } ins(%arg0 : tensor<3x4x6xf32>) -> tensor<3x24xf32>
      return %0 : tensor<3x24xf32>
    }

    func.func @trt_shuffle_reshape_copy(%arg0: tensor<1x10x20x40xf32>) -> tensor<1x10x800xf32> {
      %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2, 3>,
        reshape = array<i64: 0, 0, 800>,
        second_transpose = array<i64: 0, 1, 2>
      } ins(%arg0 : tensor<1x10x20x40xf32>) -> tensor<1x10x800xf32>
      return %0 : tensor<1x10x800xf32>
    }
    ```
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, I32, F16, F32, BF16, TensorRT_F8]>:$input,
    Optional<1DTensorOf<[I32]>>:$dynamic_reshape,
    DenseI64ArrayAttr:$first_transpose,
    OptionalAttr<DenseI64ArrayAttr>:$reshape,
    DenseI64ArrayAttr:$second_transpose,
    DefaultValuedAttr<BoolAttr, "true">:$zero_is_placeholder
  );
  let results = (outs
    AnyRankedTensor:$result  );
  let assemblyFormat = "attr-dict `ins` `(` operands `:` type(operands) `)` `->` type(results)";

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let builders = [
    // Builder for a pure static reshape operation.
    OpBuilder<(ins "TensorType":$resultType, "TypedValue<RankedTensorType>":$input,
          "bool":$zeroIsPlaceholder)>,
    // Builder for a pure dynamic reshape operation.
    OpBuilder<(ins "TensorType":$resultType, "TypedValue<RankedTensorType>":$input,
          "TypedValue<RankedTensorType>":$reshape, "bool":$zeroIsPlaceholder)>
  ];

  let extraClassDeclaration = [{
    /// Return the type of the input after the first transpose
    /// is applied.
    RankedTensorType getIntermediateType();

    /// Return the first transpose as an AffineMap permutation. If the
    /// first transpose is empty (for scalar input), then returns an empty AffineMap.
    AffineMap getFirstTransposeMap();

    /// Return the second transpose as an AffineMap permutation. If the
    /// second transpose is empty (for scalar input), then returns an empty AffineMap.
    AffineMap getSecondTransposeMap();
  }];

  let trtLayerAdd = [{
    // Silence warnings from unused generated args.
    (void)$first_transpose;
    (void)$second_transpose;

    nvinfer1::IShuffleLayer *layer = $net->addShuffle(*$input);
    layer->setZeroIsPlaceholder($zero_is_placeholder);
    layer->setFirstTranspose(getNvInferPermutation($op.getFirstTranspose()));
    if ($dynamic_reshape)
      layer->setInput(1, *$dynamic_reshape);
    if ($reshape)
      layer->setReshapeDimensions(*$reshape);
    layer->setSecondTranspose(getNvInferPermutation($op.getSecondTranspose()));
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// DeconvolutionOp
//===----------------------------------------------------------------------===//

def TensorRT_DeconvolutionOp : TensorRT_Op<"deconvolution",
      [Pure, SameOperandsAndResultElementType, AttrSizedOperandSegments,
      TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT deconvolution (IDeconvolutionLayer) operation";
  let description = [{
    The `tensorrt.deconvolution` operation represents a 2D or 3D deconvolution:
    `input`'s spatial dimensions are convolved with the filters of `kernelWeights`
    to form the result.

    The `input` is assumed to be organized such that the dimensions of the shape
    correspond to `[batch_dim, chan_dim, spatial dimensions... ]`.

    The `kernelWeights` is assumed to be organized such that the dimensions of the shape
    correspond to `[chan_dim, num_filters / num_groups, filter spatial dimensions... ]`.

    The `kernelWeights` input must be computable at build time via constant folding.

    An optional `biasWeights` is supported, which adds a per-channel constant to each
    value in the output. If supplied `biasWeights` must be the result of a
    `tensorrt.constant` operation.

    TODO: add other padding modes.
    TODO: add stride/padding/dilation/groups description.
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, F32, F16]>:$input,
    Optional<TensorRT_RankedTensorOf<[TensorRT_I8, F32, F16]>>:$kernelWeights,
    Optional<TensorRT_RankedTensorOf<[TensorRT_I8, F32, F16]>>:$biasWeights,
    OptionalAttr<ElementsAttr>:$kernelWeightsStatic,
    OptionalAttr<ElementsAttr>:$biasWeightsStatic,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pre_padding,
    DenseI64ArrayAttr:$post_padding,
    DefaultValuedAttr<UI32Attr, "1">:$num_groups,
    OptionalAttr<DenseI64ArrayAttr>:$dilation
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    attr-dict  `in` `(` $input `:` type($input) `)`
      (`kernelWeights` `(` $kernelWeights^ `:` type($kernelWeights) `)` )?
      (`biasWeights`   `(` $biasWeights^   `:` type($biasWeights)   `)` )?
      `->` type($result)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Return the number of spatial dimensions in the kernel.
    int64_t getNumSpatialDims() {
      return getType().getRank()-2;
    }

    ArrayRef<int64_t> getKernelSpatialShape() {
      return this->getKernelWeightsStatic().has_value()
            ? this->getKernelWeightsStatic()->getShapedType().getShape().take_back(
                  this->getNumSpatialDims())
            : this->getKernelWeights().getType().getShape().take_back(
                  this->getNumSpatialDims());
    }
  }] # baseClassDeclaration;
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$input, "OpFoldResult":$kernelWeights, "OpFoldResult":$biasWeights,
                   "ArrayRef<int64_t>":$stride,
                   "ArrayRef<int64_t>":$pre_padding,
                   "ArrayRef<int64_t>":$post_padding,
                   "uint32_t":$num_groups,
                   "std::optional<ArrayRef<int64_t>>":$dilation)>
  ];
  let trtLayerAdd = [{
    int32_t numOutputMaps = $op.getType().getDimSize(1);
    nvinfer1::IDeconvolutionLayer *layer = $net->addDeconvolutionNd(
        *$input, numOutputMaps,
        getNvInferDims($op.getKernelSpatialShape()),
        *$kernelWeightsStatic, *$biasWeightsStatic);
    if ($kernelWeights)
      layer->setInput(1, *$kernelWeights);
    if ($biasWeights)
      layer->setInput(2, *$biasWeights);
    layer->setStrideNd($stride);
    layer->setPrePadding($pre_padding);
    layer->setPostPadding($post_padding);
    if ($dilation)
      layer->setDilationNd(*$dilation);
    layer->setNbGroups($num_groups);
    if (!$e.isStronglyTyped()){
      FailureOr<nvinfer1::DataType> outputTrtType = getNvInferDataType($op.getLoc(),
                                                          $op.getType().getElementType());
      if (failed(outputTrtType))
        return failure();
      layer->setOutputType(0, *outputTrtType);
    }
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

def TensorRT_QuantizeOp : TensorRT_Op<"quantize",
    [Pure,
    AllShapesMatch<["input", "result"]>]>{
  let summary = "TensorRT Quantize(IQuantizeLayer) operation";
  let description = [{
    The `tensorrt.quantize` operation quantize a floating-point input tensor into
    an 8-bit signed integer output tensor using the `scale` input.

    NOTE: Although `zero_point` is an optional input for TensorRT Quantize(IQuantizeLayer)
    operation, it is removed here because it's just a placeholder which is not
    actually used and cannot be set to other values other than 0 in TensorRT.
    `zero_point` is omitted for the following decryption.

    Quantization can either be applied uniformly for the entire input ("per-tensor" manner)
    using a single scalar "scale" value, or it can be applied in a "per-channel" manner,
    meaning that different slices of the input tensor along a specified `axis` are
    quantized using different parameters in a 1D "scale" tensor, or it can be applied
    in a "2D block" manner using a 2D "scale" tensor (supports only `INT4` 2D weight inputs).

    #### Caveats

    - `scale` must be compile time constants (after constant folding by either
        MLIR-TRT or TRT itself).

    #### Argument Constraints

    If `axis` is not given:

    - `scale` must be 2D for INT4 block quantization. Block quantization always happens
    along `axis` 0.
    - `scale` must be a scalar (0D tensor i.e. `tensor<[element type]>`) for per-tensor
    quantization.

    If `axis` is given:

    - `scale` must be 1D tensors with `input.shape[axis]`
        number of elements, for per-channel quantization.
    - Similarly, the size of input dimension `input.shape[axis]` must be
      a compile time constant (i.e. it cannot be `?`).
    - `axis` must be in the range of `[0, R)`.

    #### Quantization Computation

    Let `(i_0, ..., i_{R-1})` represent indices in the index space of the
    `input` tensor, which is of rank `R`.

    In the below, the `round` function rounds f32 values to i8 values with the
    semantic of
    ["round-to-nearest-with-ties-to-even"](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).

    Per-tensor quantization transforms the input
    uniformly as follows:

    ```
    output[ i_0, ..., i_{R-1} ] :=
      clamp(
        round(
          input[ i_0, ..., i_{R-1} ] / (f32)scale
        ),
        min = (i8) -128,
        max = (i8) 127
      )
    ```

    Per-channel quantization is supported only for weight inputs. Activations
    can't be quantized per-channel. Given a 1D tensor of `scale` values and
    `axis`, the input is transformed as follows:

    ```
    output[i_0, ..., i_{axis}, ..., i_{R-1} ] :=
      clamp(
        round(
          input[ i_0, ..., i_{axis}, ..., i_{R-1} ] /
          scale[ i_{axis} ]
        ),
        min = (i8) -128,
        max = (i8) 127
      )
    ```

    Block quantization is supported only for 2-D weight inputs of type
    `INT4`. Axis for block quantization is always 0. Let `B` be the block size.
    Block quantization transforms the input as follows:

    ```
    output[i_0, i_1] :=
      clamp(
        round(
          input[ i_0, i_1 ] / scale[ i_0 // B, i_1 ]
        ),
        min = (i4) -8,
        max = (i4) 7
      )
    ```

    The output shape is the same as the input shape. The quantization axis is
    in reference to the input tensor's dimensions.

    Valid conversions are as follows:

    ```
    (f32 | f16 | bf16) -> (int8 | f8E4M3FN | int4)
    ```

    #### Examples

    ```mlir
    %1 = tensorrt.quantize in(%arg0 : tensor<10x10xf32>)
          scale(%arg1 :  tensor<1xf32>) -> tensor<10x10xi8>
    %2 = tensorrt.quantize { axis = 1 : i32 } in(%arg0 : tensor<10x10xf32>)
          scale(%arg1 : tensor<10xf32>) -> tensor<10x10xi8>
    ```
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[F32, F16, BF16]>:$input,
    TensorRankOf<[F32, F16, BF16], [0, 1, 2]>:$scale,
    OptionalAttr<I32Attr>:$axis
  );
  let results = (outs TensorRT_RankedTensorOf<[TensorRT_I8, TensorRT_F8, TensorRT_I4]>:$result);
  let assemblyFormat = [{
    attr-dict `in` `(` $input `:` type($input) `)`
    `scale` `(` $scale `:` type($scale) `)`
    `->` type($result)
  }];
  let hasVerifier = 1;

  let trtLayerAdd = [{
    nvinfer1::IQuantizeLayer *quantizeLayer{nullptr};
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(10, 0, 0)
    assert($op.getScale().getType().getRank() < 2);
#endif
    FailureOr<nvinfer1::DataType> inputTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(inputTrtType))
      return failure();
    if (!$e.isStronglyTyped()) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(9, 1, 0)
      nvinfer1::IIdentityLayer *scaleCastLayer = $net->addIdentity(*$scale);
      scaleCastLayer->setOutputType(0, nvinfer1::DataType::kFLOAT);
      $e.setMetadata(scaleCastLayer, $op);
      quantizeLayer = $net->addQuantize(*$input, *scaleCastLayer->getOutput(0));
#else
      quantizeLayer = $net->addQuantize(*$input, *$scale);
#endif
    } else {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
      quantizeLayer = $net->addQuantize(*$input, *$scale,
          *inputTrtType);
#else
      llvm_unreachable("strongly typed is enabled when compiled with TRT version < 9.1; "
        "this should never occur");
#endif
    }
    if ($axis)
      quantizeLayer->setAxis(*$axis);
    if (!$e.isStronglyTyped()) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
      quantizeLayer->setOutputType(0, *inputTrtType);
      $results.push_back(quantizeLayer->getOutput(0));
#else
      nvinfer1::IIdentityLayer *identityLayer = $net->addIdentity(*quantizeLayer->getOutput(0));
      identityLayer->setOutputType(0, *inputTrtType);
      $results.push_back(identityLayer->getOutput(0));
      $e.setMetadata(identityLayer, $op);
#endif
    } else {
      $results.push_back(quantizeLayer->getOutput(0));
    }
  }];
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

def TensorRT_DequantizeOp : TensorRT_Op<"dequantize",
    [Pure, AllShapesMatch<["input", "result"]>]>{
  let summary = "TensorRT Dequantize(IDequantizeLayer) operation";
    let description = [{
    The `tensorrt.dequantize` operation dequantize an 8-bit signed integer into
    a floating-point output tensor using the `scale` input.

    NOTE: Although `zero_point` is an optional input for TensorRT
    Dequantize(IDequantizeLayer) operation, it is removed here because it's just
    a placeholder which is not actually used and cannot be set to other values
    other than 0 in TensorRT.
    `zero_point` is omitted for the following description.

    Dequantization can either be applied uniformly for the entire input
    ("per-tensor" manner) using a single scalar "scale" value, or it can be
    applied in a "per-channel" manner, meaning that different slices of the input
    tensor along a specified `axis` are dequantized using different parameters in
    a 1D "scale" tensor, or it can be applied in a "2D block" manner using a 2D
    "scale" tensor (supports only `INT4` 2D weight inputs).

    #### Caveats

    - `scale` must be compile time constants (after constant folding by either
        MLIR-TRT or TRT itself).

    #### Argument Constraints

    If `axis` is not given:

    - `scale` must be 2D for INT4 block dequantization. Block dequantization always
    happens along `axis` 0.
    - `scale` must be a scalar (0D tensor i.e. `tensor<[element type]>`) for
    per-tensor dequantization.

    If `axis` is given:

    - `scale` must be 1D tensors with `input.shape[axis]`
        number of elements, for per-channel dequantization.
    - Similarly, the size of input dimension `input.shape[axis]` must be
      a compile time constant (i.e. it cannot be `?`).
    - `axis` must be in the range of `[0, R)`.

    #### Dequantization Computation

    Let `(i_0, ..., i_{R-1})` represent indices in the index space of the
    `input` tensor, which is of rank `R`.

    Per-tensor dequantization transforms the input
    uniformly as follows:

    ```
    output[ i_0, ..., i_{R-1} ] := input[ i_0, ..., i_{R-1} ] * (f32)scale
    ```

    Per-channel dequantization is supported only for weight inputs. Activations
    can't be dequantized per-channel. Given a 1D tensor of `scale` values and
    `axis`, the input is transformed as follows:

    ```
    output[i_0, ..., i_{axis}, ..., i_{R-1} ] :=
          input[ i_0, ..., i_{axis}, ..., i_{R-1} ] * scale[ i_{axis} ]
    ```

    Block quantization is supported only for 2-D weight inputs of type `INT4`.
    Axis for block quantization is always 0. Let `B` be the block size. Block
    quantization transforms the input as follows:

    ```
    output[i_0, i_1] :=
      input[ i_0, i_1 ] * scale[ i_0 // B, i_1 ]
    ```

    The output shape is the same as the input shape. The dequantization axis is
    in reference to the input tensor's dimensions.

    Valid conversions are as follows:

    ```
    (int8 | f8E4M3FN | int4) -> (f32 | f16 | bf16)
    ```

    #### Examples

    ```mlir
    %1 = tensorrt.dequantize in(%arg0 : tensor<10x10xi8>)
          scale(%arg1 :  tensor<1xf32>) -> tensor<10x10xf32>
    %2 = tensorrt.dequantize { axis = 1 : i32 } in(%arg0 : tensor<10x10xi8>)
          scale(%arg1 : tensor<10xf32>) -> tensor<10x10xf32>
    ```
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, TensorRT_F8, TensorRT_I4]>:$input,
    TensorRankOf<[F32, F16, BF16], [0, 1, 2]>:$scale,
    OptionalAttr<I32Attr>:$axis
  );
  let results = (outs TensorRT_RankedTensorOf<[F32, F16, BF16]>:$result);
  let assemblyFormat = [{
    attr-dict `in` `(` $input `:` type($input) `)`
    `scale` `(` $scale `:` type($scale) `)`
    `->` type($result)
  }];
  let hasVerifier = 1;
  let trtLayerAdd = [{
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(10, 0, 0)
    assert($op.getScale().getType().getRank() < 2);
#endif
    nvinfer1::ILayer *dequantizeLayer{nullptr};
    FailureOr<nvinfer1::DataType> resultTrtType = getNvInferDataType($op.getLoc(),
                                                        $op.getType().getElementType());
    if (failed(resultTrtType))
      return failure();
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(9, 1, 0)
    nvinfer1::IIdentityLayer *scaleCastLayer = $net->addIdentity(*$scale);
    scaleCastLayer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    $e.setMetadata(scaleCastLayer, $op);
    dequantizeLayer = $e.addDequantizeLayer($input, scaleCastLayer->getOutput(0),
    *resultTrtType, $axis);
#else
    dequantizeLayer = $e.addDequantizeLayer($input, $scale,
    *resultTrtType, $axis);
#endif
    FailureOr<nvinfer1::DataType> inputTrtType = getNvInferDataType($op.getLoc(),
                                                      $op.getInput().getType().getElementType());
    if (failed(inputTrtType))
      return failure();
    if (!$e.isStronglyTyped()) {
      dequantizeLayer->setPrecision(*inputTrtType);
    }
    $results.push_back(dequantizeLayer->getOutput(0));
    $e.setMetadata(dequantizeLayer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// TensorRT Dialect Extension Operations
//
// Below this point are operations "extension" operations that are not part of
// the TensorRT operation set exactly. They are either decomposed to more
// primitive operations during the `--tensorrt-expand-ops` pass or they are
// elided through canonicalization or another transformation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
def TensorRT_TransposeOp : TensorRT_Op<"transpose",
    [Pure, TensorRTInferTensorResultTypes]> {
  let summary = "Permutes the dimensions of the input tensor";

  let description = [{
    Permutes the dimensions of the `input` according to `permutation`.

    This operation is a TensorRT dialect extension operation and is lowered
    into a TensorRT shuffle operation.
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    AffineMapAttr:$permutation
  );

  let results = (outs TensorRT_Tensor:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// ExpandRankOp
//===----------------------------------------------------------------------===//
def TensorRT_ExpandRankOp : TensorRT_Op<"expand_rank",
    [Pure, SameOperandsAndResultElementType]> {

  let summary = "Expands the rank of the input tensor by inserting 1's in the shape";

  let description = [{
    The shape of the tensor is "expanded" by inserting 1's into the shape.

    This operation is typically used to ensure that result tensor meets the required
    rank for broadcasting in elementwise operations or for certain operations
    such as matrix multiplication.

    This operation is a TensorRT dialect extension operation and is lowered
    into a TensorRT shuffle operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";

  let builders = [
      OpBuilder<(ins "Type":$result, "Value":$input, "ArrayRef<ReassociationIndices>":$reassociation)>
    ];

  let extraClassDeclaration = [{
    // The below operations support interop with upstream
    // templatized reshape transforms:

    /// Return the input operand.
    Value getSrc() {
      return getInput();
    }
    /// Return the type of the input operand.
    ShapedType getSrcType() {
      return cast<ShapedType>(getInput().getType());
    }
    /// Return the type of the result.
    ShapedType getResultType() {
      return cast<ShapedType>(getResult().getType());
    }
    /// Return the reassociation dims derived from the
    /// input and result shapes.
    SmallVector<ReassociationIndices> getReassociationIndices();
  }] # baseClassDeclaration;
}

//===----------------------------------------------------------------------===//
// CollapseRankOp
//===----------------------------------------------------------------------===//
def TensorRT_CollapseRankOp : TensorRT_Op<"collapse_rank",
    [Pure, SameOperandsAndResultElementType]> {

  let summary = "Collapses the rank of the input tensor by removing 1's from the shape";

  let description = [{
    The shape of the tensor is "collapsed" by removing 1's (not necessarily all
    of them). This operation achieves the result commonly known as "squeezing"
    a tensor in deep learning frameworks.

    This operation is a TensorRT dialect extension operation and is lowered
    into a TensorRT shuffle operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";

  let extraClassDeclaration = [{
    /// Return the indices in the input type shape that correspond to removed
    /// unit dimensions.
    SmallVector<int64_t> getInputShapeDimIndicesOfRemovedDims();

    // The below operations support interop with upstream
    // templatized reshape transforms:

    /// Return the input operand.
    Value getSrc() {
      return getInput();
    }
    /// Return the type of the input operand.
    ShapedType getSrcType() {
      return cast<ShapedType>(getInput().getType());
    }
    /// Return the type of the result.
    ShapedType getResultType() {
      return cast<ShapedType>(getResult().getType());
    }
    /// Return the reassociation dims derived from the
    /// input and result shapes.
    SmallVector<ReassociationIndices> getReassociationIndices();
  }] # baseClassDeclaration;

  let builders = [
    OpBuilder<(ins "Type":$result, "Value":$input, "ArrayRef<ReassociationIndices>":$reassociation)>
  ];
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def TensorRT_BroadcastOp : TensorRT_Op<"broadcast",
      [Pure, TensorRTPartiallyInferTensorResultTypes]> {
  let summary = "Represents a broadcast explicitly that occurs in different TensorRT operations";

  let description = [{
    `tensorrt.broadcast` is an explicit representation of the broadcasting
    operation that occurs in TensorRT operations such as
    `tensorrt.element_wise`. In addition, this broadcast operation is allowed to
    perform a rank expansion as well. The semantics are aligned with the
    StableHLO `broadcast_in_dim` operation.

    The `broadcast_dims` attribute contains a list of indices that such that
    input type dimension `i` is mapped to `broadcast_dims[i]` of the result
    type. `broadcast_dims` do not have to be increasing; the mapping is allowed
    to permute the ordering of the input dimensions.

    If the result shape is not fully specified, then the `shape` tensor
    containing must be provided to specify the result shape.

    This operation is a TensorRT dialect extension operation and will either be
    canonicalized away (by absorbing into e.g. `tensorrt.element_wise`) or be lowered
    into a reshape operation such as `tensorrt.shuffle`.

    #### Example

    ```mlir
    %result = tensorrt.broadcast %input broadcast_dims<3, 1>
                : tensor<1x10xf32> to tensor <4x10x4x28xf32>
    ```
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    Optional<TensorRT_ShapeTensor>:$shape,
    DenseI64ArrayAttr:$broadcast_dims
  );

  let results = (outs TensorRT_Tensor:$result);

  let builders = [
    // Static builder.
    OpBuilder<(ins "Type":$result, "Value":$input, "ArrayRef<int64_t>":$broadcast_dims)>,
  ];

  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = [{
    $input `broadcast_dims` `<` custom<StaticIndexI64Array>($broadcast_dims) `>`
      (`shape` `(` $shape^ `:` type($shape) `)`)?
      attr-dict `:` type($input) `to` type($result)
  }];

  let extraClassDeclaration = [{
    /// Returns the permutation of the input shape caused by the
    /// broadcast dims order. For example, broadcast dims [3, 1, 2] would
    /// return [1, 2, 0].
    AffineMap getBroadcastDimsPermutation();
  }];
}

//===----------------------------------------------------------------------===//
// ArgMinOp / ArgMaxOp
//===----------------------------------------------------------------------===//

class TensorRT_ArgMinMaxOp<string mnemonic> :
    TensorRT_Op<mnemonic, [Pure, TensorRTInferTensorResultTypes,
                          AllRanksMatch<["input", "values", "indices"]>,
                          AllShapesMatch<["values", "indices"]>]> {
  let arguments = (ins
    TensorRT_Tensor:$input,
    I64Attr:$axis
  );
  let results = (outs TensorRT_Tensor:$values, TensorRT_RankedTensorOf<[I32]>:$indices);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict $input `:` type($input) `->` type(results)";

  let builders = [
    // Builder with automatically derived type. The result type is never
    // rank-reduced.
    OpBuilder<(ins "Value":$input, "int64_t":$axis)>
  ];
}

def TensorRT_ArgMinOp : TensorRT_ArgMinMaxOp<"argmin"> {
  let summary = "Represents an argmin operation";
  let description = [{
    The `tensorrt.argmin` operation returns the minimum value as well as
    the index for the that value along the specified dimension. The
    result types are not rank-reduced.

    This operation is a TensorRT dialect extension operation and will be
    lowered to a `tensorrt.top_k` operation before translation to a
    TensorRT engine.
  }];
}

def TensorRT_ArgMaxOp : TensorRT_ArgMinMaxOp<"argmax"> {
  let summary = "Represents an argmax operation";
  let description = [{
    The `tensorrt.argmax` operation returns the maximum value as well as
    the index for the that value along the specified dimension. The
    result types are not rank-reduced.

    This operation is a TensorRT dialect extension operation and will be
    lowered to a `tensorrt.top_k` operation before translation to a
    TensorRT engine.
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//
def TensorRT_YieldOp : TensorRT_Op<"yield",
    [Pure, ReturnLike, Terminator, ParentOneOf<[
      "tensorrt::IfOp", "tensorrt::ForOp", "tensorrt::WhileOp",
      "tensorrt::OpaquePluginOp"]>]> {

  let summary = "Yields one or more SSA values from a tensorrt dialect region";

  let description = [{
    The `tensorrt.yield` operation yields SSA values from tensorrt dialect op region
    and terminate the regions. The semantics of how the values are yielded is
    defined by the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def TensorRT_ReshapeOp : TensorRT_Op<"reshape",
      [Pure, AllElementTypesMatch<["input", "result"]>]> {
  let summary = "A reshape operation";
  let description = [{
    The `tensorrt.reshape` operation performs a reshape operation. The reshape
    can be dynamic in the sense that the sizes of the dimensions are unknown,
    but all ranks must be known. Reshapes that only add/remove unit dimensions
    will be canonicalized to `tensorrt.expand_rank` and `tensorrt.collapse_rank`
    respectively.

    If the dynamic `shape` operand is not provided, then the result type is
    allowed to have at most one dynamic dimensin. Otherwise, the `shape`
    operand must be provided.

    #### Examples

    A static reshape:

    ```
    %reshaped = tensorrt.reshape %input : tensor<10xf32> to tensor<2x5xf32>
    ```

    A dynamic reshape:

    ```
    %reshaped = tensorrt.reshape %input shape(%shape: tensor<2xi32>)
        : tensor<?xf32> to tensor<?x?xf32>
    ```

    A dynamic reshape with only one unknown dimension:

    ```
    %reshaped = tensorrt.reshape %input
        : tensor<?xf32> to tensor<2x?xf32>
    ```
  }];

  let arguments = (ins
    TensorRT_Tensor:$input,
    Optional<1DTensorOf<[I32]>>:$shape
  );
  let results = (outs TensorRT_Tensor:$result);

  // Note: the `` after "$shape^" is to remove the extra space between the SSA
  // shape value and the colon. So it prints like `shape($shape: tensor<2xi32>)`
  // instead of `shape($shape : tensor<2xi32>)`.
  let assemblyFormat = [{
    $input (`shape` `(` $shape^ `` `:` type($shape) `)`)? attr-dict
      `:` type($input) `to` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let builders = [
    // Construct reshape with static shape.
    OpBuilder<(ins "Type":$result, "Value":$input)>
  ];
}


//===----------------------------------------------------------------------===//
// ResizeNearestOp
//===----------------------------------------------------------------------===//

def TensorRT_ResizeNearestOp : TensorRT_Op<"resize_nearest", [Pure,
 TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT Resize(IResizeLayer with NEAREST mode) operation";
  let description = [{

    The `tensorrt.resize_nearest` operation resize an input tensor into an output
    tensor with the NEAREST mode. The operation resizes innermost `m` dimensions
    of an input tensor of rank N, when 0 <= m <= min(3, N).

    For IResizeLayer there are 2 parameters `shape` and `scales` which are used to
    determine output shape. If the input has static input/output shapes, then these
    2 parameters don't need to be kept as attributes since `tensorrt.resize_nearest`
    can directly retrieve the output shape from its output type information to
    initialize `shape` parameter. If the input has dynamic shapes, then `scales`
    parameter is used as an attribute to determine the output shapes.

    `tensorrt.resize_nearest` uses `coordinate_transformation` to control how the
    coordinates in the output tensor are mapped to coordinates in the input tensor.
    Specifically, for each spatial dimension i, input[..., f(y_i), ...] = result[..., y_i, ...]
    where f is given by one of the following equations, depending on the value of
    the `coordinate_transformation` attribute:

        -   ResizeCoordinateTransformation::kALIGN_CORNERS, in this mode:
            f(y_i) = y_i * (shape(input)[i] - 1) / (shape(result)[i] - 1)
        -   ResizeCoordinateTransformation::kASYMMETRIC, in this mode:
            f(y_i) = y_i * shape(input)[i] / shape(result)[i]
        -   ResizeCoordinateTransformation::kHALF_PIXEL, in this mode:
            f(y_i) = (y_i + 0.5) * shape(input)[i] / shape(result)[i] - 0.5

    `tensorrt.resize_nearest` uses `nearest_rounding` to control the rounding mode for
    nearest neighbor resize, it indicates how to get "nearest" pixel from the calculated
    input coordinate f(y_i) above, it supports following modes:

        -   ResizeRoundMode::kHALF_UP, round half up
        -   ResizeRoundMode::kHALF_DOWN, round half down
        -   ResizeRoundMode::kFLOOR, round to floor
        -   ResizeRoundMode::kCEIL, round to ceil

    There is another parameter `selector_for_single_pixel` for IResizeLayer which
    controls selector when resize to single pixel output, it supports 2 modes:

        -   FORMULA, use formula to map the original index, f(y_i) above
        -   UPPER, use the upper left pixel

  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$input,
    Optional<TensorRT_ShapeTensor>:$output_shape,
    OptionalAttr<DenseF32ArrayAttr>:$scales,
    TensorRT_ResizeCoordinateTransformationAttr:$coordinateTransformation,
    TensorRT_ResizeRoundModeAttr:$nearestRounding,
    TensorRT_ResizeSelectorAttr:$selectorForSinglePixel
  );
  let results = (outs
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$result);
  let assemblyFormat = "attr-dict $input ( `,` $output_shape^ )? `:` functional-type(operands,results) ";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::IResizeLayer *layer = network->addResize(*$input);
    if ($op.getOutputShape())
      layer->setInput(1, *$output_shape);
    else if ($op.getScales().has_value())
      layer->setScales($scales.value().begin(), $scales.value().size());
    else
      layer->setOutputDimensions(getNvInferDims($op.getType().getShape()));
    layer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    layer->setCoordinateTransformation(*$coordinateTransformation);
    layer->setSelectorForSinglePixel(*$selectorForSinglePixel);
    layer->setNearestRounding(*$nearestRounding);
    $results.push_back(layer->getOutput(0));
  }];
}

//===----------------------------------------------------------------------===//
// ResizeLinearOp
//===----------------------------------------------------------------------===//

def TensorRT_ResizeLinearOp : TensorRT_Op<"resize_linear", [Pure,
 TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT Resize(IResizeLayer with LINEAR mode) operation";
  let description = [{

    The `tensorrt.resize_linear` operation resize an input tensor into an output
    tensor with the LINEAR mode. The operation resizes innermost `m` dimensions
    of an input tensor of rank N, when 0 <= m <= min(3, N).

    For IResizeLayer there are 2 parameters `shape` and `scales` which are used to
    determine output shape. If the input has static input/output shapes, then these
    2 parameters don't need to be kept as attributes since `tensorrt.resize_linear`
    can directly retrieve the output shape from its output type information to
    initialize `shape` parameter. If the input has dynamic shapes, then `scales`
    parameter is used as an attribute to determine the output shapes.

    `tensorrt.resize_linear` uses `coordinate_transformation` to control how the
    coordinates in the output tensor are mapped to coordinates in the input tensor.
    Specifically, for each spatial dimension i, input[..., f(y_i), ...] = result[..., y_i, ...]
    where f is given by one of the following equations, depending on the value of
    the `coordinate_transformation` attribute:

        -   ResizeCoordinateTransformation::kALIGN_CORNERS, in this mode:
            f(y_i) = y_i * (shape(input)[i] - 1) / (shape(result)[i] - 1)
        -   ResizeCoordinateTransformation::kASYMMETRIC, in this mode:
            f(y_i) = y_i * shape(input)[i] / shape(result)[i]
        -   ResizeCoordinateTransformation::kHALF_PIXEL, in this mode:
            f(y_i) = (y_i + 0.5) * shape(input)[i] / shape(result)[i] - 0.5

    There is another parameter `selector_for_single_pixel` for IResizeLayer which
     controls selector when resize to single pixel output, it supports 2 modes:

        -   FORMULA, use formula to map the original index
        -   UPPER, use the upper left pixel

  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$input,
    Optional<TensorRT_ShapeTensor>:$output_shape,
    OptionalAttr<DenseF32ArrayAttr>:$scales,
    TensorRT_ResizeCoordinateTransformationAttr:$coordinateTransformation,
    TensorRT_ResizeSelectorAttr:$selectorForSinglePixel
  );
  let results = (outs
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$result);
  let assemblyFormat = "attr-dict $input ( `,` $output_shape^ )? `:` functional-type(operands,results) ";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::IResizeLayer *layer = network->addResize(*$input);
    if ($op.getOutputShape())
      layer->setInput(1, *$output_shape);
    else if ($op.getScales().has_value())
      layer->setScales($scales.value().begin(), $scales.value().size());
    else
      layer->setOutputDimensions(getNvInferDims($op.getType().getShape()));
    layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
    layer->setCoordinateTransformation(*$coordinateTransformation);
    layer->setSelectorForSinglePixel(*$selectorForSinglePixel);
    $results.push_back(layer->getOutput(0));
  }];
}  // def TensorRT_ResizeLinearOp : TensorRT_Op<"resize_linear", []>

//===----------------------------------------------------------------------===//
// ResizeCubicOp
//===----------------------------------------------------------------------===//

def TensorRT_ResizeCubicOp : TensorRT_Op<"resize_cubic", [Pure,
 TensorRTPartiallyInferTensorResultTypes]>{
  let summary = "TensorRT Resize(IResizeLayer with CUBIC mode) operation";
  let description = [{

    The `tensorrt.resize_cubic` operation resize an input tensor into an output
    tensor with the LINEAR mode. The operation resizes innermost `m` dimensions
    of an input tensor of rank N, when 0 <= m <= min(3, N).

    For IResizeLayer there are 2 parameters `shape` and `scales` which are used to
    determine output shape. If the input has static input/output shapes, then these
    2 parameters don't need to be kept as attributes since `tensorrt.resize_cubic`
    can directly retrieve the output shape from its output type information to
    initialize `shape` parameter. If the input has dynamic shapes, then `scales`
    parameter is used as an attribute to determine the output shapes.

    `tensorrt.resize_cubic` uses `coordinate_transformation` to control how the
    coordinates in the output tensor are mapped to coordinates in the input tensor.
    Specifically, for each spatial dimension i, input[..., f(y_i), ...] = result[..., y_i, ...]
    where f is given by one of the following equations, depending on the value of
    the `coordinate_transformation` attribute:

        -   ResizeCoordinateTransformation::kALIGN_CORNERS, in this mode:
            f(y_i) = y_i * (shape(input)[i] - 1) / (shape(result)[i] - 1)
        -   ResizeCoordinateTransformation::kASYMMETRIC, in this mode:
            f(y_i) = y_i * shape(input)[i] / shape(result)[i]
        -   ResizeCoordinateTransformation::kHALF_PIXEL, in this mode:
            f(y_i) = (y_i + 0.5) * shape(input)[i] / shape(result)[i] - 0.5

    There is another parameter `selector_for_single_pixel` for IResizeLayer which
     controls selector when resize to single pixel output, it supports 2 modes:

        -   FORMULA, use formula to map the original index
        -   UPPER, use the upper left pixel

    `tensorrt.resize_cubic` uses `cubic_coeff` as coefficient in cubic interpolation.
    Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
  }];

  let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$input,
    Optional<TensorRT_ShapeTensor>:$output_shape,
    OptionalAttr<DenseF32ArrayAttr>:$scales,
    TensorRT_ResizeCoordinateTransformationAttr:$coordinateTransformation,
    TensorRT_ResizeSelectorAttr:$selectorForSinglePixel,
    F32Attr:$cubicCoeff
  );
  let results = (outs
    TensorRT_RankedTensorOf<[TensorRT_I8, F16, F32]>:$result);
  let assemblyFormat = "attr-dict $input ( `,` $output_shape^ )? `:` functional-type(operands,results) ";
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::IResizeLayer *layer = network->addResize(*$input);
    if ($op.getOutputShape())
      layer->setInput(1, *$output_shape);
    else if ($op.getScales().has_value())
      layer->setScales($scales.value().begin(), $scales.value().size());
    else
      layer->setOutputDimensions(getNvInferDims($op.getType().getShape()));
    layer->setResizeMode(nvinfer1::ResizeMode::kCUBIC);
    layer->setCoordinateTransformation(*$coordinateTransformation);
    layer->setSelectorForSinglePixel(*$selectorForSinglePixel);
    layer->setCubicCoeff($cubicCoeff);
    $results.push_back(layer->getOutput(0));
  }];
}

//===----------------------------------------------------------------------===//
// OpaquePluginOp
//===----------------------------------------------------------------------===//

def TensorRT_OpaquePluginOp : TensorRT_Op<"opaque_plugin", [
      Pure, IsolatedFromAbove,
      DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]>{
  let summary = "a representation of an opaque external TensorRT plugin";

  let description = [{
    The `tensorrt.opaque_plugin` operation represents an external TensorRT
    plugin that is provided as a dynamic shared object library (DSO).

    This operation supports the PluginV3/PluginCreatorV3One API introduced
    in TensorRT 10. Older TensorRT plugin APIs are not supported.

    The attributes of the plugin specify the path to the DSO as well as the
    information needed to construct a plugin instance using the plugin's
    PluginCreator implementation. The `creator_params` are serialized into
    a PluginFieldCollection in order to

    The steps for loading and creating the plugin are executed as follows.
    First, a reference to the PluginCreatorV3One for the TensorRT plugin
    must be acquired. There are three possible mechanisms for doing this:

    1. If both the `dso_path` and the `creator_func` attributes are provided,
    then the DSO is loaded (e.g. on POSIX using `dlopen`) and the symbol name
    specified by the attribute `creator_func` is searched
    for in the loaded library. This must be a function that accepts no
    parameters and returns an `nvinfer1::IPluginCreatorV3One*`. In the
    event of an error in the function, it is allowed to return `nullptr`.
    The compiler takes ownership over the creator and caches it for all
    translations of this operation that have the same `plugin_name`, so
    the creator class should be stateless.

    2. If the `dso_path` is specified, but no `creator_func` is provided,
    then the DSO is loaded and it is expected that loading the DSO
    triggers registration of the creator with TensorRT's global plugin creator
    registry. The plugin creator is then looked up in that registry.

    3. Otherwise, if no `dso_path` is given, then it is expected that
    that the plugin creator is already registered in the glboal plugin
    registry, and we lookup the plugin by name/version/namespace in the
    global registry.

    After acquiring a reference to the creator, we create the plugin with
    the following steps:

    1. The plugin creator is instantiated by calling the function with name
       specified by `creator_func` and is cache. Subsequent
       translations of an operation with the same `plugin_name` will not
       create a new PluginCreator.

    2. The translator queries the instantiated PluginCreator for the relevant
       fields by calling the `PluginCreatorV3One::getFieldNames()` member
       function. The plugin must provide an implementation for `getFieldNames`.
       All attributes in `creator_params` are validated against the plugin
       specifications returned by `getFieldNames()`.

    3. The `creator_params` are serialized into a `PluginFieldCollection`
       and passed to the creator's `getPlugin` call.


    #### Dynamic Shapes and Shape Calculation Region

    The plugin has a single region, `shapes_region`, which is allowed to be
    empty.

    When populated, the `shapes_region` should contain a single block of
    scalar arithmetic IR for calculating result shapes. The block arguments should
    contain one scalar argument for each input dimension for each tensor operand.
    The single block should be terminated by `tensorrt.yield` and yield a scalar
    for each dimension for each result tensor.

    The `shapes_region` is made optional even when the results have dynamic
    extents in order to enable lazy population of the content by directly
    querying the actual TensorRT plugin.  The `tensorrt-infer-plugin-shapes`
    will load the plugin and use a simple mechanism to construct the right IR
    in the shapes region.

    #### TODO:

    The following items are currently unimplemnted:

    1. Any sort of verification by loading the plugin before translation
       time.

  }];

  let arguments = (ins
    StrAttr:$plugin_name,
    StrAttr:$plugin_version,
    StrAttr:$plugin_namespace,
    OptionalAttr<StrAttr>:$dso_path,
    OptionalAttr<StrAttr>:$creator_func,
    DictionaryAttr:$creator_params,
    Variadic<TensorRT_Tensor>:$inputs);

  let results = (outs Variadic<TensorRT_Tensor>:$results);

  let regions = (region AnyRegion:$shapes_region);

  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Infer a set of shape components from the shapes region, if present.
    std::optional<SmallVector<ShapedTypeComponents>> inferShapeComponentsFromShapesRegion();
  }];

  let assemblyFormat = [{
    attr-dict  `(` ($inputs^)? `)` `:` functional-type($inputs, $results)
    ( $shapes_region^ )?
  }];

  let trtLayerAdd = [{
    FailureOr<nvinfer1::ILayer*> layer = $e.addOpaquePlugin(
        $op, $results);
    if(failed(layer))
      return failure();
  }];
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def TensorRT_ScatterOp : TensorRT_Op<"scatter_nd",
      [Pure, AllElementTypesMatch<["data", "updates"]>, TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT scatter (IScatterLayer with kND mode) operation";
  let description = [{
    The `tensorrt.scatter_nd` operation creates an output tensor by copying values
    from an input tensor `data` and then updating values by the given `indices`
    and `updates` tensors.

    This operation takes three inputs:

    - A `data` tensor of rank r >= 1, that stores the values to be duplicated in
    the output.
    - An `indices` tensor of rank q >= 1 that determines which locations in the
    output to write new data to. The indices are interpreted as a tensor of
    rank q-1 of indexing tuples.
    - An `updates` tensor of rank s, that provides the data to write to output,
    at the index specified by the corresponding location in the `indices`
    tensor. For this operation, rank s = (q - 1) + (r - indices.shape[-1]).

    This operation produces one output:
    - An `output` tensor of the same dimensions as data that stores the
    resulting values of the transformation.

    The output of the operation is obtained by updating the data tensor's value
    to values specified by updates at specific index positions specified by
    indices. Its shape is the same as the shape of data.

    The types of data, update, and output shall be the same.

    This op follows the semantics of
    [tf.scatter_nd](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add)
    and
    [onnx.ScatterND](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND)
    and
    [torch.index_select](https://pytorch.org/docs/stable/generated/torch.scatter.html)

    #### Output Computation:

    Assuming that data dims are {d_0,...,d_{r-1}} and indices dims are
    {i_0,...,i_{q-1}}, define k = indices.shape[q-1], the updates dims must be
    {i_0,...,i_{q-2},d_k,...,d_{r-1}}.

    The updating can be computed by:
        foreach slice in indices[i_0,...i_{q-2} ]
            output[indices[slice]] = updates[slice]

    #### Caveats
    Writes to the same output element cause undefined behavior.
  }];

   let arguments = (ins
    TensorRT_RankedTensorOf<[I32, F16, F32, BF16]>:$data,
    TensorRT_RankedTensorOf<[I32]>:$indices,
    TensorRT_RankedTensorOf<[I32, F16, F32, BF16]>:$updates
  );
  let results = (outs TensorRT_Tensor:$result);
  let assemblyFormat = [{
    attr-dict
    `data` `(` $data `:` type($data) `)`
    `indices` `(` $indices `:` type($indices) `)`
    `updates` `(` $updates `:` type($updates) `)`
  }];
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::IScatterLayer *layer = $net->addScatter(
        *$data, *$indices, *$updates, nvinfer1::ScatterMode::kND);
    layer->setMode(nvinfer1::ScatterMode::kND);
    nvinfer1::IIdentityLayer *identityLayer = $net->addIdentity(*layer->getOutput(0));
    FailureOr<nvinfer1::DataType> inputTrtType = getNvInferDataType($op.getLoc(),
                                                      $op.getType().getElementType());
    if (failed(inputTrtType))
      return failure();
    if (!$e.isStronglyTyped())
      identityLayer->setOutputType(0, *inputTrtType);
    $results.push_back(identityLayer->getOutput(0));
    $e.setMetadata(layer, $op);
    $e.setMetadata(identityLayer, $op);
  }];
}

//===----------------------------------------------------------------------===//
// ScatterElementsOp
//===----------------------------------------------------------------------===//

def TensorRT_ScatterElementsOp : TensorRT_Op<"scatter_elements",
      [Pure, AllElementTypesMatch<["data", "updates"]>,
      AllShapesMatch<["indices", "updates"]>,
      AllRanksMatch<["data", "updates", "indices"]>,
      TensorRTInferTensorResultTypes]>{
  let summary = "TensorRT scatter (IScatterLayer with kELEMENT mode) operation";
  let description = [{
    The `tensorrt.scatter_elements` operation creates an output tensor by
    copying values from an input tensor `data` and then updating values at
    indices given by the `indices` tensor with values from the `updates` tensor.
    Its output shape is the same as the shape of `data`.

    This operation takes three inputs:

    - A `data` tensor of rank r >= 1, that stores the values to be duplicated in
    the output. - An `indices` tensor of the same rank as `data` that determines
    which locations in the output to write new data to. The indices are
    interpreted as a tensor of rank 1 of indexing tuples. - An `updates` tensor
    of the same rank as `data`, that provides the data to write to output, at
    the index specified by the corresponding location in the `indices` tensor. -
    An optional `axis` attribute that identifies an axis of `data` along which
    to scatter on. By default, this is the outermost axis, which is 0.

    This operation produces one output: - An `output` tensor of the same
    dimensions as `data` that stores the resulting values of the transformation.

    This op follows the semantics of
    [tf.scatter_nd](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add)
    and
    [onnx.ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND)
    and
    [torch.index_select](https://pytorch.org/docs/stable/generated/torch.scatter.html)

    #### Output Computation:

    For each entry in `updates`, the target index in `data` is obtained by
    combining the corresponding entry in `indices` with the index of the entry
    itself: the index-value for dimension = axis is obtained from the value of
    the corresponding entry in `indices` and the index-value for dimension !=
    axis is obtained from the index of the entry itself.

    For each element X of indices:
      Let J denote a sequence for the subscripts of X
      Let K = sequence J with element[axis] replaced by X
      Then,
        output[K] = updates[J]

    For example, if indices has dimensions [N,C,H,W] and axis is 2, then the
    updates happen as:

    for n in [0, N)
        for c in [0, C)
            for h in [0, H)
                for w in [0, W)
                    output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]


    #### Caveats

    Writes to the same output element cause undefined behavior.
  }];

   let arguments = (ins
    TensorRT_RankedTensorOf<[TensorRT_I8, I32, F16, F32]>:$data,
    TensorRT_RankedTensorOf<[I32]>:$indices,
    TensorRT_RankedTensorOf<[TensorRT_I8, I32, F16, F32]>:$updates,
    OptionalAttr<I64Attr>:$axis
  );
  let results = (outs TensorRT_Tensor:$result);
  let assemblyFormat = [{
    attr-dict
    `data` `(` $data `:` type($data) `)`
    `indices` `(` $indices `:` type($indices) `)`
    `updates` `(` $updates `:` type($updates) `)`
  }];
  let hasVerifier = 1;
  let trtLayerAdd = [{
    nvinfer1::IScatterLayer *layer = $net->addScatter(
        *$data, *$indices, *$updates, nvinfer1::ScatterMode::kELEMENT);
    layer->setMode(nvinfer1::ScatterMode::kELEMENT);
    if ($axis)
      layer->setAxis(*$axis);
    $results.push_back(layer->getOutput(0));
    $e.setMetadata(layer, $op);
  }];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTOPS_TD
