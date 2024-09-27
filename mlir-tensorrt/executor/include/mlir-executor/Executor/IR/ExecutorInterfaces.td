#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORINTERFACES_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORINTERFACES_TD

include "mlir/IR/OpBase.td"

def RuntimeBuiltinInterface : OpInterface<"RuntimeBuiltinInterface"> {
  let description = [{
    Interface for operations that are likely to be lowered to an external
    function call for the target runtime/target. This interface does not
    mean that the op has to be lowered to a function call, only that it
    is likely.

    It provides a way for the operation to specify a naming scheme for the
    operation when lowered to a function call. This is provided because
    Executor IR does not provide a way to lower all attributes and types to
    values. For example, a cast operation `executor.extf` from f16 to f32
    would be lowered to "call @_extf_f16_f32" because we must dispatch
    extensions of different types to different underlying runtime functions.
    This is used instead of encoding the input value as some sort of variant
    type (which Executor IR does not provide) and passing the types encoded
    as values.

    Building on the `extf` example, if you are lowering to a target which has a
    native operation for `extf` builtin, then you can translate to that operation
    instead of lowering to function call by setting the appropriate flags on
    the `convert-executor-to-executor` pass.
  }];

  let cppNamespace = "::mlir::executor";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the types to encode in the operation suffix. These are
        only allowed to be scalar types.
      }],
      /*retTy=*/"SmallVector<Type>",
      /*methodName=*/"getTypesForNameSuffix",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector($_op->getResultTypes());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the name of the runtime builtin function",
      /*retTy=*/"FailureOr<std::string>",
      /*methodName=*/"getRuntimeBuiltinFunctionName",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Type> suffixTypes = $_op.getTypesForNameSuffix();
        return ::mlir::executor::detail::getRuntimeBuiltinFunctionNameImpl($_op,
          suffixTypes, dataLayout);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the function type of the runtime builtin",
      /*retTy=*/"executor::ExecutorFunctionType",
      /*methodName=*/"getRuntimeBuiltinFunctionType",
      /*args=*/(ins "TypeRange":$results, "TypeRange":$operands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return mlir::executor::ExecutorFunctionType::get(
            $_op.getContext(),
            llvm::to_vector(operands),
            llvm::to_vector(results), mlir::UnitAttr());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Creates the equivalent `executor.call` to an opaque function, but does
        not replace the operation. Replacement is done by the callee.
      }],
      /*retTy=*/"FailureOr<::mlir::CallOpInterface>",
      /*methodName=*/"lowerToCall",
      /*args=*/(ins "::mlir::ArrayRef<Value>":$operands,
                    "::mlir::RewriterBase &":$rewriter,
                    "::mlir::ModuleOp":$moduleOp,
                    "const ::mlir::TypeConverter &":$typeConverter,
                    "const ::mlir::DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::executor::detail::lowerToCallDefaultImpl(
          $_op, operands, moduleOp, rewriter, typeConverter, dataLayout);
      }]
    >
  ];

  let verify = [{
    ::mlir::DataLayout dataLayout = ::mlir::DataLayout::closest($_op);
    return detail::verifyRuntimeBuiltinInterface($_op, dataLayout);
  }];
}

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORINTERFACES_TD
