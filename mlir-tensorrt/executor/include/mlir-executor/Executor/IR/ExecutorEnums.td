#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORENUMS_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORENUMS_TD

include "mlir-executor/Executor/IR/ExecutorDialect.td"

include "mlir/IR/EnumAttr.td"

class Executor_I32EnumAttr<string name, string description,
  list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::executor";
  let genSpecializedAttr = 0;
}

class Executor_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic, list<Trait> traits = []> :
  EnumAttr<Executor_Dialect, intEnumAttr, mnemonic, traits> {
  let assemblyFormat = "`<` $value `>`";
}

def Executor_ICmpType : Executor_I32EnumAttr<
  "ICmpType",
  "types of integer comparison",
  [
    I32EnumAttrCase<"eq", 0>,
    I32EnumAttrCase<"ne", 1>,
    I32EnumAttrCase<"slt", 2>,
    I32EnumAttrCase<"sgt", 3>,
    I32EnumAttrCase<"sle", 4>,
    I32EnumAttrCase<"sge", 5>,
    I32EnumAttrCase<"ult", 6>,
    I32EnumAttrCase<"ugt", 7>,
  ]
>;

def Executor_ICmpTypeAttr : Executor_EnumAttr<Executor_ICmpType, "icmp_type">;

def Executor_FCmpType : Executor_I32EnumAttr<
  "FCmpType",
  "types of integer comparison",
  [
    I32EnumAttrCase<"_false", 0>,
    I32EnumAttrCase<"oeq",    1>,
    I32EnumAttrCase<"ogt",    2>,
    I32EnumAttrCase<"oge",    3>,
    I32EnumAttrCase<"olt",    4>,
    I32EnumAttrCase<"ole",    5>,
    I32EnumAttrCase<"one",    6>,
    I32EnumAttrCase<"ord",    7>,
    I32EnumAttrCase<"ueq",    8>,
    I32EnumAttrCase<"ugt",    9>,
    I32EnumAttrCase<"uge",    10>,
    I32EnumAttrCase<"ult",    11>,
    I32EnumAttrCase<"ule",    12>,
    I32EnumAttrCase<"une",    13>,
    I32EnumAttrCase<"uno",    14>,
    I32EnumAttrCase<"_true",  15>
  ]
>;
def Executor_FCmpTypeAttr : Executor_EnumAttr<Executor_FCmpType, "fcmp_type">;


def Executor_CallingConvention : Executor_I32EnumAttr<
  "CallingConvention",
  "the convention used to call a function",
  [
    // Arguments are passed as normal args
    I32EnumAttrCase<"unpacked", 0>,
    // Arguments are packed into an aggregate and then unpacked
    // in the function. This is used to work around argument limits
    // in specific backends like Lua.
    I32EnumAttrCase<"packed", 1>
  ]
>;
def Executor_CallingConventionAttr : Executor_EnumAttr<Executor_CallingConvention, "calling_convention">;

// Declare the type of memory. This is used in the memref memory space
// field to distinguish between device, host, and page-locked memory.
class Executor_MemoryTypeCase<string name, int value> : I32EnumAttrCase<name, value>;

def Executor_MemoryTypeHost : Executor_MemoryTypeCase<"host", 0>;
def Executor_MemoryTypeHostPinned : Executor_MemoryTypeCase<"host_pinned", 1>;
def Executor_MemoryTypeDevice : Executor_MemoryTypeCase<"device", 2>;
def Executor_MemoryTypeUnified : Executor_MemoryTypeCase<"unified", 3>;
def Executor_MemoryTypeRaw :Executor_MemoryTypeCase<"raw", 4>;

def Executor_MemoryType : Executor_I32EnumAttr<
  "MemoryType",
  "types of memory",
  [
    Executor_MemoryTypeHost,
    Executor_MemoryTypeHostPinned,
    Executor_MemoryTypeDevice,
    Executor_MemoryTypeUnified,
    Executor_MemoryTypeRaw
  ]
>;

def Executor_MemoryTypeAttr : Executor_EnumAttr<Executor_MemoryType, "memory_type">;


def Executor_ArgABIKind : Executor_I32EnumAttr<
  "ArgABIKind",
  "the ABI of an argument of a function",
  [
    I32EnumAttrCase<"byval", 0>,
    I32EnumAttrCase<"byref", 1>,
  ]
>;
def Executor_ArgABIKindAttr : Executor_EnumAttr<Executor_ArgABIKind, "abi">;


#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORENUMS_TD
