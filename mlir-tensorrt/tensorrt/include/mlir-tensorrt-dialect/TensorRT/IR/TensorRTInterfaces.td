#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTINTERFACES
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTINTERFACES

include "mlir/IR/OpBase.td"

def TensorRTOpInterface : OpInterface<"TensorRTOpInterface"> {
  let description = [{
    This interface is to be implemented by operations that represent
    TensorRT layers.
  }];

  let cppNamespace = "::mlir::tensorrt";

  let methods = [];
}

def TensorRTEnumAttrInterface : AttrInterface<"TensorRTEnumAttrInterface"> {
  let cppNamespace = "::mlir::tensorrt";

  let methods = [
    InterfaceMethod<[{
      Retrieve the nvinfer1 namespace equivalent enum value as a
      string.
    }], "std::string", "getNvInferEnumValueStr", (ins), [{
      return $_self.getNvInferValueStr();
    }]>
  ];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTINTERFACES
