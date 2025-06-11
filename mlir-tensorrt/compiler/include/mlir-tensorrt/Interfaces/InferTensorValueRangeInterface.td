#ifndef MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE
#define MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InferTensorValueRangeInterface
//===----------------------------------------------------------------------===//

def InferTensorValueRangeInterface : OpInterface<"InferTensorValueRangeInterface"> {
  let description = [{
    Allows operations to participate in range analysis for tensor values by
    providing a methods that allows them to specify lower and upper bounds on their
    result(s) given lower and upper bounds on their input(s) if known.
  }];
  let cppNamespace = "::mlirtrt::compiler";

  let methods = [
    InterfaceMethod<[{
      Infer the bounds on the results of this op given the lattice representation
      of the bounds for its arguments. For each result value or block argument
      (that isn't a branch argument, since the dataflow analysis handles
      those case), the method should call `setValueRange` with that `Value`
      as an argument. When implemented, `setValueRange` should be called on
      all result values for the operation.

      This method allows for more precise implementations when operations
      want to reason about inputs which may be undefined during the analysis.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRangesFromOptional",
    /*args=*/(ins "::llvm::ArrayRef<::mlirtrt::compiler::IntOrTensorValueRange>":$argRanges,
                  "::mlirtrt::compiler::SetTensorValueLatticeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{}]>,
  ];
}


#endif // MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE
