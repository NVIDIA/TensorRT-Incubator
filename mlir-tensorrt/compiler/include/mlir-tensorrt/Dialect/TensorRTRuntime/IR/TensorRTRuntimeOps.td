#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTRUNTIMEOPS_TD
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTRUNTIMEOPS_TD

include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeDialect.td"
include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeTypes.td"
include "mlir-tensorrt/Dialect/CUDA/IR/CUDATypes.td"
include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def TensorRTRuntime_TensorRTRuntimeOpTrait: NativeOpTrait<"TensorRTRuntimeOpTrait"> {
  let cppNamespace = "::mlir::trtrt";
}

class TensorRTRuntime_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorRTRuntime_Dialect, mnemonic, !listconcat(traits, [TensorRTRuntime_TensorRTRuntimeOpTrait])> {
}

//===----------------------------------------------------------------------===//
// CreateRuntimeOp
//===----------------------------------------------------------------------===//

def TensorRTRuntime_CreateRuntimeOp : TensorRTRuntime_Op<
      "create_runtime", [Pure]> {

  let description = [{
    Returns a handle to a `!trtrt.runtime` (`nvinfer::IRuntime` object).
  }];

  let arguments = (ins );
  let results = (outs TensorRTRuntime_Runtime:$result);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CompileOp
//===----------------------------------------------------------------------===//

def TensorRTRuntime_CompileOp : TensorRTRuntime_Op<"compile", [Pure]> {

  let description = [{
    Compiles a `func.func` representing a TensorRT network into a serialized
    TensorRT engine.
  }];

  let arguments = (ins SymbolRefAttr:$trt_func);
  let results = (outs TensorRTRuntime_Context:$result);
  let assemblyFormat = [{
    attr-dict $trt_func `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// EnqueueOp
//===----------------------------------------------------------------------===//

def TensorRTRuntime_EnqueueOp : TensorRTRuntime_Op<"enqueue", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<TensorKindOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AttrSizedOperandSegments,
    DestinationStyleOpInterface
]> {
  let description = [{

    Asynchronously executes the computation represented by the
    `execution_context` on the specified CUDA stream. This operation
    is a bufferizable destination-passing-style (DPS) operation.

    This means that the `inputs` and `outputs` can accept either
    all `tensor` types or all `memref` types. If the types are `tensor`
    types, then the the values passed to the `outs` parameter must
    be equal in type and number to the operation's results.

    When the `inputs` and `outputs` are `memref` types, then the
    operation should have no results.

    The `host_tensor_args` attribute is a list of indices into the
    `inputs` list indicating which arguments should be host tensors.
  }];

  let arguments = (ins TensorRTRuntime_Context:$execution_context,
                       CUDA_Stream:$stream,
                       Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outs,
                       OptionalAttr<DenseI64ArrayAttr>:$host_tensor_args);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $execution_context `stream` `(` $stream `)` ` `
    (`host_tensor_args` $host_tensor_args^ ` ` )?
    `(` $inputs `)` `outs` `(` $outs `)`
    attr-dict `:` functional-type($inputs, $outs)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Declare the outs as inits/outs to DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutsMutable(); }

    /// Return true if the operand is a host tensor argument.
    bool isOperandOnHost(OpOperand *operand) {
      unsigned operandIdx = operand->getOperandNumber();
      if(std::optional<ArrayRef<int64_t>> indices = getHostTensorArgs()) {
        return llvm::is_contained(*indices, operandIdx - 2);
      }
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// EnqueueAllocOp
//===----------------------------------------------------------------------===//

def TensorRTRuntime_EnqueueAllocOp : TensorRTRuntime_Op<"enqueue_alloc", [
    DeclareOpInterfaceMethods<TensorKindOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let description = [{
    Asynchronously executes the computation represented by the
    `execution_context` on the specified CUDA stream. This operation
    can accept inputs of either tensor or memref types and returns
    results of either tensor or memref types.
  }];

  let arguments = (ins 
    TensorRTRuntime_Context:$execution_context,
    CUDA_Stream:$stream,
    Variadic<AnyTypeOf<[AnyMemRef, AnyTensor]>>:$inputs,
    OptionalAttr<DenseI64ArrayAttr>:$host_tensor_args
  );

  let results = (outs Variadic<AnyTypeOf<[AnyMemRef, AnyTensor]>>:$results);

  let assemblyFormat = [{
    $execution_context `stream` `(` $stream `)` ` `
    (`host_tensor_args` $host_tensor_args^ ` ` )?
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, $results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return true if the operand is a host tensor argument.
    bool isOperandOnHost(OpOperand *operand) {
      unsigned operandIdx = operand->getOperandNumber();
      if(std::optional<ArrayRef<int64_t>> indices = getHostTensorArgs()) {
        return llvm::is_contained(*indices, operandIdx - 2);
      }
      return false;
    }
  }];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTRUNTIMEOPS_TD
