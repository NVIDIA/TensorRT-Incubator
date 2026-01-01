#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORTYPES_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORTYPES_TD

include "mlir-executor/Executor/IR/ExecutorDialect.td"
include "mlir-executor/Executor/IR/ExecutorEnums.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Executor_Type<string name, string typeMnemonic,
                    list<Trait> traits = []> : TypeDef<Executor_Dialect, name,
                                                       traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//
def Executor_Ptr : TypeDef<Executor_Dialect, "Pointer", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
]>{
  let mnemonic = "ptr";
  let description = [{
    `executor.ptr` represents an opaque pointer.
  }];

  let parameters = (ins EnumParameter<Executor_MemoryType>:$address_space);
  let assemblyFormat = [{
    `<` $address_space `>`
  }];
}

class Executor_PtrOf<Executor_MemoryTypeCase enumcase> : Type<
      And<[CPred<"llvm::isa<mlir::executor::PointerType>($_self)">,
           CPred<
            "llvm::cast<mlir::executor::PointerType>($_self).getAddressSpace()"
            "== mlir::executor::MemoryType::"#enumcase.symbol>]>,
      "pointer of type " # enumcase.str>;
def Executor_HostPtr : Executor_PtrOf<Executor_MemoryTypeHost>,
   BuildableType<"$_builder.getType<mlir::executor::PointerType>(mlir::executor::MemoryType::host)">;
def Executor_HostPinnedPtr : Executor_PtrOf<Executor_MemoryTypeHostPinned>;
def Executor_HostVisiblePtr : AnyTypeOf<[
  Executor_PtrOf<Executor_MemoryTypeHostPinned>,
  Executor_PtrOf<Executor_MemoryTypeHost>
]>;
def Executor_DevicePtr : Executor_PtrOf<Executor_MemoryTypeDevice>;

//===----------------------------------------------------------------------===//
// StrLiteralType
//===----------------------------------------------------------------------===//

def Executor_StrLiteral : TypeDef<Executor_Dialect, "StrLiteral", []> {
  let mnemonic = "str_literal";
  let description = [{
    Represents a string literal.
  }];
}

//===----------------------------------------------------------------------===//
// TableType
//===----------------------------------------------------------------------===//
def Executor_Table : TypeDef<Executor_Dialect, "Table", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>
]>{
  let mnemonic = "table";
  let description = [{
    Represents a generic table/struct whose body contains a list of elements.
    The table's elements are accessed by integer index in the order that they
    are listed. The number and type of elements in the body cannot be modified
    after the table is created via the `executor.table.create` operation.

    The body types may be any valid executor integer, float, `!executor.ptr` or
    nested tables.

    The `!executor.table` type associated operations could be lowered to LLVM
    structs or arrays types, depending on the types in the body. The Lua
    reference lowering can lower directly to Lua tables or some other optimized
    representation.

a  }];

  let parameters = (ins ArrayRefParameter<"Type">:$body);

  let assemblyFormat = [{
    `<` $body `>`
  }];
}

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//
def Executor_ExecutorFunctionType : TypeDef<Executor_Dialect, "ExecutorFunction", []> {
  let mnemonic = "func";
  let description = [{
    Represents a function signature. The signature is allowed to have
    a variadic number of arguments that appears after arguments that are
    not variadic. The number and type of returns must be fixed.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$args,
                        ArrayRefParameter<"Type">:$results,
                        OptionalParameter<"UnitAttr">:$trailingVarArg);

  let assemblyFormat = [{
    `<` `(` custom<VarArgEllipses>($args, $trailingVarArg) `)`  `->` custom<FuncResults>($results) `>`
  }];

  let extraClassDeclaration = [{
    /// Clone the function with new inputs/result types. This is required
    /// to interact with FunctionOpInterface.
    ExecutorFunctionType clone(TypeRange inputs, TypeRange results);
  }];
}

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORTYPES_TD
