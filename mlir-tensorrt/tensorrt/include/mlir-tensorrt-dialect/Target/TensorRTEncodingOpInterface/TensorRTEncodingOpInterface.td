#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTTRANSLATIONINTERFACE_TD
#define MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTTRANSLATIONINTERFACE_TD

include "mlir/IR/OpBase.td"

def TensorRTEncodingOpInterface : OpInterface<"TensorRTEncodingOpInterface"> {
  let description = [{
    This interface is to be implemented by operations that represent
    TensorRT layers.
  }];

  let cppNamespace = "::mlir::tensorrt";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Encodes an operation using the TensorRT network encoder.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"encodeOp",
      /*args=*/(ins "NvInferNetworkEncoder&":$encoder,
                    "SmallVector<::nvinfer1::ITensor*>&":$results),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->emitOpError("translation to TensorRT network unimplemented");
      }]
    >,
  ];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTTRANSLATIONINTERFACE_TD
