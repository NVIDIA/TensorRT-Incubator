#ifndef MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE

include "mlir/IR/OpBase.td"

def ToLoopsOpInterface : OpInterface<"ToLoopsOpInterface"> {
  let description = "Interface for lowering to loops";

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Lower the operation to a loop nest. Returns
        the outermost loop that should replace the original
        op, but does not actually perform the replacement.
      }],
        "::mlir::FailureOr<::mlir::LowerToLoopsResult>",
        "lowerToLoops",
        (ins "::mlir::RewriterBase&":$rewriter),
        "",
        [{
          llvm_unreachable("Not implemented");
        }]
    >
  ];
}

#endif // MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE
