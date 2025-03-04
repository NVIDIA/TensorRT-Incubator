#ifndef MLIR_TENSORRT_INTERFACES_BUFFERIZATIONSCOPEINTERFACE
#define MLIR_TENSORRT_INTERFACES_BUFFERIZATIONSCOPEINTERFACE

include "mlir/IR/OpBase.td"

def BufferizationScopeOpInterface : OpInterface<"BufferizationScopeOpInterface"> {
  let description = [{
    An operation interface that allows module-like operations to declare options
    which should be used for bufferization.
  }];
  let cppNamespace = "::mtc";

  let methods = [
    InterfaceMethod<[{
        Returns the options which should be used for bufferization.
      }],
        "std::optional<::mlir::bufferization::OneShotBufferizationOptions>",
        "getBufferizationOptions"
    >,
    InterfaceMethod<[{
        Perform post-bufferization cleanup actions.
      }],
        "::mlir::LogicalResult",
        "performPostBufferizationActions",
        (ins "::mlir::IRRewriter&":$rewriter)
    >
  ];
}

#endif // MLIR_TENSORRT_INTERFACES_BUFFERIZATIONSCOPEINTERFACE
