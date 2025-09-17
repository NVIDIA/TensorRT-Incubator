#ifndef MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORBASE_TD
#define MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORBASE_TD

include "mlir-executor/Executor/IR/ExecutorAttributes.td"
include "mlir-executor/Executor/IR/ExecutorDialect.td"
include "mlir-executor/Executor/IR/ExecutorTypes.td"
include "mlir-executor/Executor/IR/ExecutorInterfaces.td"

/// A normal Executor op
class Executor_Op<
  string mnemonic,
  list<Trait> traits = []
> : Op<
  Executor_Dialect,
  mnemonic,
  !listconcat(traits, [])
>;

def I4  : I<4>;
defvar Executor_ArithFloatTypes = [F64, F32, F16, F8E4M3FN, BF16, F4E2M1FN];
defvar Executor_ArithIntTypes = [I64, I32, I16, I8, I4, Index];

// Integer signness semantics are specified by individual operations, not
// the integer type.
def Executor_Integer : AnyTypeOf<!listconcat(Executor_ArithIntTypes,
      [I1])>;
def Executor_Float : AnyTypeOf<Executor_ArithFloatTypes>;
def Executor_MemRefElementTypes: AnyTypeOf<
  !listconcat(Executor_ArithFloatTypes, [I32, I16, I8, I1])>;
def Executor_AnyType : AnyTypeOf<[
      Executor_Integer, Executor_Float, Executor_Table,
      Executor_Ptr]>;
def Executor_Index : AnyTypeOf<[I64, I32]>;

#endif // MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORBASE_TD
