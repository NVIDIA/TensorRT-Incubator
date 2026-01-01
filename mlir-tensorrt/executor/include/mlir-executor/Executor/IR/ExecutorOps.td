#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOROPS_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOROPS_TD


include "mlir-executor/Executor/IR/ExecutorBase.td"
include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

class IntOrFloatElementBitWidth<string name> :
    StrFunc<"$" # name # ".getType().getIntOrFloatBitWidth()">;

class AllBitWidthsMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, IntOrFloatElementBitWidth<"_self">.result,
                              "bit width">;

//===----------------------------------------------------------------------===//
// Shorthand declarations
//===----------------------------------------------------------------------===//

/// Base class for cast-like operations.
class Executor_CastOp<string mnemonic, list<Trait> traits = []>
  : Executor_Op<mnemonic, !listconcat(traits, [
            Pure,
            DeclareOpInterfaceMethods<RuntimeBuiltinInterface,
              ["getTypesForNameSuffix"]>
          ])> {
  let assemblyFormat = "$operand attr-dict `:` type($operand) `to` type($result)";
  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getTypesForNameSuffix() {
      return SmallVector<Type>{getOperand().getType(), getType()};
    }
  }];
}

/// Base class for ops that cast from one float type to another.
class Executor_FloatCastOp<string mnemonic, list<Trait> traits = []>
        : Executor_CastOp<mnemonic, traits> {
  let arguments = (ins Executor_Float:$operand);
  let results = (outs Executor_Float:$result);
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def Executor_AssertOp : Executor_Op<"assert", [
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Assert operation with message attribute";
  let description = [{

    The `executor.assert` operation represents a runtime assertion with a single
    boolean operand and an error message attribute.

    If the argument is `true` this operation has no effect. Otherwise, the
    program execution will abort. The provided error message may be used by a
    runtime to propagate the error to the user.

    Example:

    ```mlir
    executor.assert %b, "Expected some condition to be true"
    ```
  }];

  let arguments = (ins I1:$arg, StrAttr:$msg);

  let assemblyFormat = "$arg `,` $msg attr-dict";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Executor_ConstantOp : Executor_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]> {

  let summary = "integer or floating point constant";
  let description = [{
    The `executor.constant` operation produces an SSA value equal to some
    integer or floating-point constant specified by an attribute.

    This is basically the same as `arith.constant`, but we are more limited
    in terms of allowed types based on the target VM.

    For example, if Lua is the target VM, then depending on Lua's build
    config, integers are either signed 32 bit or signed 64 bit.

    Example:

    ```
    // Integer constant
    %1 = executor.constant 42 : i32

    // Equivalent generic form
    %1 = "executor.constant"() {value = 42 : i32} : () -> i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);

  let results = (outs AnyTypeOf<[Executor_Integer, Executor_Float]>:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$type), [{
      build($_builder, $_state, cast<TypedAttr>(value));
    }]>,
  ];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DataSegmentOp
//===----------------------------------------------------------------------===//

def Executor_DataSegmentOp : Executor_Op<"data_segment", [Symbol]> {

  let summary = "declares a read-only constant resource";

  let description = [{
    The `executor.data_segment` operation represents a declaration of a region
    of memory.

    The region of memory must have an initial `value` attached. If the
    `uninitialized` attribute is set, then the `value` is used just to indicate
    the size of the zero-initialized data, and `value` should be a dense splat
    elements attribute. Only the size is encoded in the compiled artifact (no
    actual data is encoded). This lets us create a mechanism for indicating that
    globals should be initialized with a new zero-filled allocation, similar to
    the BSS section in an ELF.

    If `constant` is specified, then the memory is read-only, and this is
    exclusive to `uninitialized`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       AnyAttrOf<[ElementsAttr, StrAttr]>:$value,
                       UnitAttr:$constant,
                       UnitAttr:$uninitialized,
                       OptionalAttr<I64Attr>:$alignment,
                       DefaultValuedAttr<Executor_MemoryTypeAttr,
                         "::mlir::executor::MemoryType::host">:$address_space);
  let assemblyFormat = [{
    $sym_name attr-dict
      ( `constant` $constant^ )?
      (`uninitialized` $uninitialized^ )?
      (`align` $alignment^)?
      (`address_space` $address_space^)?
      $value
  }];

  let extraClassDeclaration = [{
    /// Construct the op from the location context.
    static DataSegmentOp create(Location loc, StringRef name,
                                Attribute value,
                                bool constant,
                                bool uninitialized,
                                IntegerAttr alignment = {});

    /// Get the element type of the initial value.
    Type getElementType() {
      if (auto elementsAttr = ::llvm::dyn_cast<ElementsAttr>(getValue()))
        return elementsAttr.getElementType();
      assert(::llvm::isa<StringAttr>(getValue()) && "expected string attribute");
      return IntegerType::get(getContext(), 8);
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StrLiteralOp
//===----------------------------------------------------------------------===//

def Executor_StrLiteralOp : Executor_Op<"str_literal", [Pure]> {
  let summary = "declares a constant string literal";
  let description = [{
    The `executor.str_literal` operation declares an SSA value
    that holds a single string literal.
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs Executor_StrLiteral:$result);
  let assemblyFormat = [{
    attr-dict $value
  }];
}

//===----------------------------------------------------------------------===//
// ConstantResourceLoadOp
//===----------------------------------------------------------------------===//

def Executor_ConstantResourceLoadOp : Executor_Op<"load_data_segment",
  [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "loads a constant resource into a new host buffer";

  let description = [{
    The `executor.load_data_segment` operation allocates a new host buffer
    and loads the specified resource into the buffer.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Executor_Ptr:$result);

  let builders = [OpBuilder<(ins "FlatSymbolRefAttr":$name), [{
    build($_builder, $_state, PointerType::get($_builder.getContext(),
            MemoryType::host), name);
  }]>];

  let assemblyFormat = "attr-dict $name `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def Executor_GlobalOp : Executor_Op<"global", [Symbol, IsolatedFromAbove]> {
  let summary = "defines a global variable";

  let description = [{
    The `executor.global` operation describes a global variable declaration and,
    optionally, the IR that initializes the global when the program is loaded.

    Providing an initialization region is optional. Besides initialization
    regions, a global with a numeric constant value can also utilize the
    `initial_value` attribute. Only one of `initial_value` and an initialization
    region can be used. If the program does not provide an initialization region
    or an initial_value, then its state is considered undefined.

    If an initialization region is provided, then it should be terminated with
    an `executor.return` statement that returns the SSA value which should be
    stored into the region. The type of this value must match the `type`
    attribute.

    The operation can optionally specify that the global is a constant. In this
    case, a `executor.set_global` referencing this global is illegal, the global
    can only be set by its initialization region.

    Please note that ordering of `executor.global` operations matter: Since any
    number of `executor.global` variables can be declared, the final backend
    should execute the initialization of global variables one after another in
    the order that they are listed in the IR. Therefore, certain globals can
    depend on the initialization of other globals as long as the global being
    used in the initialization region was initialized earlier in the program
    declaration.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<TypedAttrInterface>:$initial_value,
                       UnitAttr:$constant);
  let regions = (region AnyRegion:$body_region);
  let assemblyFormat = [{
    $sym_name (`constant` $constant^)? `:` $type attr-dict-with-keyword
    ($body_region^)?
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$name, "DenseI8ResourceElementsAttr":$initial_value,
      CArg<"bool", "true">:$constant)>,
    OpBuilder<(ins "StringRef":$name, "Type":$type,
      CArg<"std::function<void(OpBuilder&,Location)>", "{}">:$initBuilder,
      CArg<"bool", "false">:$constant)>
  ];

  let extraClassDeclaration = [{
    bool hasInitRegion() {
      return !getBodyRegion().getBlocks().empty();
    }

    Block *getInitBody() {
      assert(hasInitRegion() && "does not have init region");
      return &getBodyRegion().getBlocks().front();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def Executor_ReturnOp : Executor_Op<"return", [Pure, Terminator]> {
  let summary = "global initialization or function return operation";
  let description = [{
    The `executor.return` op represents a return within a region that yields
    some results, such as a global initialization region.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def Executor_GetGlobalOp : Executor_Op<"get_global", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let description = [{
    The `executor.get_global` operation returns an SSA value representing the
    global stored at the given symbol name.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "GlobalOp":$globalOp), [{
      build($_builder, $_state, globalOp.getType(),
        FlatSymbolRefAttr::get(globalOp));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the global that this operation references.
    GlobalOp getGlobal(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// SetGlobalOp
//===----------------------------------------------------------------------===//

def Executor_SetGlobalOp : Executor_Op<"set_global", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let description = [{
    The `executor.set_global` operation sets the value of the global
    with the given symbol name to the given `value`.
  }];

  let arguments = (ins AnyType:$value,  FlatSymbolRefAttr:$name);
  let assemblyFormat = "$value `,` $name attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// CoroYieldOp
//===----------------------------------------------------------------------===//

def Executor_CoroYieldOp : Executor_Op<"coro_yield", []> {
  let summary = "suspends a coroutine";

  let description = [{
    The 'executor.coro_yield' operation suspends the current coroutine and
    transfers control back to the program point immediately after the
    point where the coroutine was resumed.

    The control flow returns to the the 'coro_yield' point when the coroutine
    is resumed. Note that 'coro_yield' is not a "terminator", but invoking the
    coro suspension has "return-like" semantics with respect to 'executor.coro_resume`.

    Therefore, the types passed to the continuation must match the result
    types of the containing function.

    Any function containing a 'coro_yield' is a coroutine function, and while
    the compiler might not object until runtime when invoking
    it directly with 'func.call', this is invalid and will result in undefined
    behavior. A coroutine function can only be invoked through the sequence of
    'executor.coro_create' and then one or more 'executor.coro_await' calls.
  }];

  let arguments = (ins Variadic<AnyType>:$yielded);
  let assemblyFormat = "attr-dict ($yielded^ `:` type($yielded))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CoroCreateOp
//===----------------------------------------------------------------------===//

def Executor_CoroCreateOp : Executor_Op<"coro_create", [Pure]> {
  let description = [{
    Constructs a coroutine handle from a function or coroutine function.
    Functions may or may not use 'executor.coro_yield' in their bodies
    (abcense of 'coro_yield' does not prevent a 'func.func' from being
    used as a coroutine, although the in that case 'coro_await' is
    identical to just using 'func.call').
  }];
  let arguments = (ins FlatSymbolRefAttr:$func);
  let results = (outs FunctionType:$result);

  let assemblyFormat = [{
    attr-dict $func `:` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CoroAwaitOp
//===----------------------------------------------------------------------===//

def Executor_CoroAwaitOp : Executor_Op<"coro_await", [
    CallOpInterface,
    TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "::llvm::cast<FunctionType>($_self).getResults()">]> {
  let summary = "resume coroutine";

  let description = [{
    `executor.coro_await` resumes a coroutine. The 'callee_operands' must either be
    empty or they must match the operands of the coroutine (for the initial
    resume). The results yielded from the coroutine should always match
    the function signature results, so the result types of `executor.coro_await`
    must match the callee function result types.

    The results of `executor.coro_await` are a flag indicating whether or not
    the coroutine was resumed (false indicates the coroutine is terminated and cannot
    be resumed) followed by the list of values yielded from the coroutine.
    If the 'status' flag is false, then the 'results' values contain undefined
    contents.
  }];

  let arguments = (ins FunctionType:$callee,
                       Variadic<AnyType>:$callee_operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs I1:$status, Variadic<AnyType>:$results);

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
      $_state.addTypes(::llvm::cast<FunctionType>(callee.getType()).getResults());
    }]>];

  let extraClassDeclaration = [{
    // Functions for CallOpInterface

    /// Get the operand range forwarded to callee.
    operand_range getArgOperands() {
      return getCalleeOperands();
    }

    /// Get the operand range forwarded to callee.
    MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
    }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, cast<Value>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $callee_operands (`:` type($callee_operands)^)? `)` attr-dict `:` type($callee)
  }];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def Executor_FuncOp : Executor_Op<"func", [IsolatedFromAbove,
  FunctionOpInterface, AutomaticAllocationScope, CallableOpInterface]> {
  let summary = "an executor function";

  let description = [{
    The `executor.func` operation defines either a function along
    with its definition or a function declaration that is defined
    externally by runtime bindings.

    Example:

    ```mlir
    executor.func @enqueue_kernel(%arg0: i32, ...)
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttrOf<Executor_ExecutorFunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "ExecutorFunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    static executor::FuncOp create(Location location, StringRef name,
      ExecutorFunctionType type, ArrayRef<NamedAttribute> attrs = {});
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }
    ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getArgs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def Executor_CallOp : Executor_Op<"call", [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call to an Executor IR function";

  let description = [{
    The `executor.call` operation represents a procedure call to an externally
    defined function given by the `callee` symbol name. The procedure is invoked
    by passing `args`.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$args,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict
       `:`  functional-type($args, $results)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$types, "StringRef":$callee, "ValueRange":$args), [{
      build($_builder, $_state, types, callee, args, ArrayAttr{}, ArrayAttr{});
    }]>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() { return getArgs(); }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }

    MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }
  }];
}


//===----------------------------------------------------------------------===//
// Integer arithmetic ops
//===----------------------------------------------------------------------===//

// Class for arithmetic binary operations.
class Executor_ArithBinaryOpBase<Type type, string mnemonic,
                                      list<Trait> traits = []> :
        Executor_Op<mnemonic,
            !listconcat([Pure, SameOperandsAndResultType], traits)> {
  dag commonArgs = (ins type:$lhs, type:$rhs);
  let results = (outs type:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}
// Class for unary arithmetic operations.
class Executor_ArithUnaryOpBase<Type type, string mnemonic,
                            list<Trait> traits = []> :
        Executor_Op<mnemonic,
            !listconcat([Pure, SameOperandsAndResultType], traits)> {
  dag commonArgs = (ins type:$operand);
  let results = (outs type:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}
// Class for arithmetic binary operations operating on integers.
class Executor_ArithBinaryIntOp<string mnemonic,
                              list<Trait> traits=[]> :
    Executor_ArithBinaryOpBase<AnyTypeOf<Executor_ArithIntTypes>, mnemonic,
      traits> {
  let arguments = commonArgs;
}
class Executor_ArithBinaryBitwiseIntOp<string mnemonic,
                              list<Trait> traits=[]> :
    Executor_ArithBinaryOpBase<Executor_Integer, mnemonic,
      traits> {
  let arguments = commonArgs;
}
class Executor_ArithBinaryOpIntWithFolder<string mnemonic,
                              list<Trait> traits=[]> :
    Executor_ArithBinaryIntOp<mnemonic, traits> {
  let hasFolder = 1;
}
// Class for arithmetic unary operations operating on integers.
class Executor_ArithUnaryIntOp<string mnemonic,
                               list<Trait> traits = []> :
      Executor_ArithUnaryOpBase<AnyTypeOf<Executor_ArithIntTypes>, mnemonic,
                                traits> {
  let arguments = commonArgs;
}

def Executor_AddIOp
    : Executor_ArithBinaryOpIntWithFolder<"addi", [Commutative]> {
  let hasCanonicalizer = 1;
}
def Executor_SubIOp : Executor_ArithBinaryOpIntWithFolder<"subi">;
def Executor_MulIOp : Executor_ArithBinaryOpIntWithFolder<"muli", [Commutative]>;
// Round towards zero
def Executor_SDivIOp : Executor_ArithBinaryOpIntWithFolder<"sdivi">;
// Round towards -inf
def Executor_SFloorDivIOp : Executor_ArithBinaryOpIntWithFolder<"sfloor_divi">;
// Remainder
def Executor_SRemIOp : Executor_ArithBinaryOpIntWithFolder<"sremi">;
// Shift an integer value to the left by a variable amount
def Executor_ShiftLeftIOp : Executor_ArithBinaryIntOp<"shift_lefti">;
// Shift right arithmetic
def Executor_ShiftRightArithmeticIOp : Executor_ArithBinaryIntOp<"shift_right_arithmetici">;
// Shift right logical. Interger operand is reinterprested as unsigned before
// right shift is applied.
def Executor_ShiftRightLogicalIOp : Executor_ArithBinaryIntOp<"shift_right_logicali">;

def Executor_BitwiseAndIOp : Executor_ArithBinaryBitwiseIntOp<"bitwise_andi">;
def Executor_BitwiseOrIOp  : Executor_ArithBinaryBitwiseIntOp<"bitwise_ori">;
def Executor_BitwiseXOrIOp : Executor_ArithBinaryBitwiseIntOp<"bitwise_xori">;

//===----------------------------------------------------------------------===//
// Integer math ops
// These operations correspond to math dialect ops and must be offloaded to
// a function call or expanded to more trivial ops.
//===----------------------------------------------------------------------===//

def Executor_AbsIOp : Executor_ArithUnaryIntOp<"absi", [RuntimeBuiltinInterface]>;

//===----------------------------------------------------------------------===//
// Floating-point arithmetic ops
//===----------------------------------------------------------------------===//

// Class for arithmetic binary operations operating on float data type.
class Executor_ArithBinaryFloatOp<string mnemonic,
                              list<Trait> traits=[]> :
    Executor_ArithBinaryOpBase<AnyTypeOf<Executor_ArithFloatTypes>, mnemonic,
        traits> {
  let arguments = commonArgs;
}

def Executor_AddFOp : Executor_ArithBinaryFloatOp<"addf",  [Commutative]>;
def Executor_SubFOp : Executor_ArithBinaryFloatOp<"subf">;
def Executor_MulFOp : Executor_ArithBinaryFloatOp<"mulf", [Commutative]>;
def Executor_DivFOp : Executor_ArithBinaryFloatOp<"divf">;
def Executor_RemFOp : Executor_ArithBinaryFloatOp<"remf">;

//===----------------------------------------------------------------------===//
// Floating-point math ops
// These operations correspond to math dialect ops and must be offloaded to
// a function call or expanded to more trivial ops.
//===----------------------------------------------------------------------===//

// Class for arithmetic unary operations operating on float data types.
class Executor_ArithUnaryMathFloatOp<string mnemonic,
                               list<Trait> traits = []> :
      Executor_ArithUnaryOpBase<AnyTypeOf<Executor_ArithFloatTypes>, mnemonic,
                               !listconcat(traits,
                                [Pure, RuntimeBuiltinInterface])> {
  let arguments = commonArgs;
}
class Executor_ArithBinaryMathFloatOp<string mnemonic,
                               list<Trait> traits = []> :
      Executor_ArithBinaryOpBase<AnyTypeOf<Executor_ArithFloatTypes>, mnemonic,
                               !listconcat(traits,
                                [Pure, RuntimeBuiltinInterface])> {
  let arguments = commonArgs;
}


// Unary ops
def Executor_AbsFOp : Executor_ArithUnaryMathFloatOp<"absf">;
def Executor_CbrtOp : Executor_ArithUnaryMathFloatOp<"cbrt">;
def Executor_CeilOp : Executor_ArithUnaryMathFloatOp<"ceil">;
def Executor_CosOp : Executor_ArithUnaryMathFloatOp<"cos">;
def Executor_ErfOp : Executor_ArithUnaryMathFloatOp<"erf">;
def Executor_ExpOp : Executor_ArithUnaryMathFloatOp<"exp">;
def Executor_Exp2Op : Executor_ArithUnaryMathFloatOp<"exp2">;
def Executor_Expm1Op : Executor_ArithUnaryMathFloatOp<"expm1">;
def Executor_FloorOp : Executor_ArithUnaryMathFloatOp<"floor">;
def Executor_LogOp : Executor_ArithUnaryMathFloatOp<"log">;
def Executor_Log10Op : Executor_ArithUnaryMathFloatOp<"log10">;
def Executor_Log1pOp : Executor_ArithUnaryMathFloatOp<"log1p">;
def Executor_Log2Op : Executor_ArithUnaryMathFloatOp<"log2">;
def Executor_NegfOp : Executor_ArithUnaryMathFloatOp<"negf">;
def Executor_SinOp : Executor_ArithUnaryMathFloatOp<"sin">;
def Executor_SqrtOp : Executor_ArithUnaryMathFloatOp<"sqrt">;
def Executor_TanOp : Executor_ArithUnaryMathFloatOp<"tan">;
def Executor_TanhOp : Executor_ArithUnaryMathFloatOp<"tanh">;
def Executor_RoundOp : Executor_ArithUnaryMathFloatOp<"round">;

// Binary ops
def Executor_Atan2Op : Executor_ArithBinaryMathFloatOp<"atan2">;
def Executor_CopysignOp : Executor_ArithBinaryMathFloatOp<"copysign">;

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

def Executor_ICmpOp : Executor_Op<"icmp", [
      Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let description = [{
    `executor.icmp` performs a comparison between `lhs` and `rhs` integers.
    The comparison type is given by the `predicate` attribute.
  }];
  let results = (outs I1:$result);
  let arguments = (ins Executor_Integer:$lhs, Executor_Integer:$rhs,
    Executor_ICmpTypeAttr:$predicate);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// FCmpOp
//===----------------------------------------------------------------------===//

def Executor_FCmpOp : Executor_Op<"fcmp", [
      Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let description = [{
    `executor.fcmp` performs a comparison between `lhs` and `rhs` floats.
    The comparison type is given by the `predicate` attribute.
  }];
  let results = (outs I1:$result);
  let arguments = (ins Executor_Float:$lhs, Executor_Float:$rhs,
    Executor_FCmpTypeAttr:$predicate);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

def Executor_BitcastOp : Executor_Op<"bitcast",
      [Pure, AllBitWidthsMatch<["input", "result"]>]> {
  let description = [{
    `executor.bitcast` op performs a bitcast between values of equal bit width.
  }];
  let arguments = (ins AnyTypeOf<[F4E2M1FN, F8E4M3FN, F64, F32, F16, I64, I32, I16, I8, I4]>:$input);
  let results = (outs AnyTypeOf<[F4E2M1FN, F8E4M3FN, F64, F32, F16, I64, I32, I16, I8, I4]>:$result);
  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def Executor_SelectOp : Executor_Op<"select",
      [Pure, AllTypesMatch<["true_value", "false_value", "result"]>]> {
  let description = [{
    `executor.select`op chooses one of the `true_value` or `false_value` based on
    value of the predicate. If the value of predicate is 1, `true_value` is chosen
    otherwise `false_value` is chosen.
  }];
  let arguments = (ins I1:$predicate, Executor_AnyType:$true_value,
                                      Executor_AnyType:$false_value);
  let results = (outs Executor_AnyType:$result);
  let assemblyFormat = "attr-dict $predicate `,` $true_value `,` $false_value `:` type($true_value)";
}

//===----------------------------------------------------------------------===//
// ExtfOp
//===----------------------------------------------------------------------===//

def Executor_ExtfOp : Executor_FloatCastOp<"extf"> {
  let description = [{
    Cast a floating point type to another floating point type of a wider
    bitwidth. The numeric values should be equivalent.
  }];
}

//===----------------------------------------------------------------------===//
// Truncf
//===----------------------------------------------------------------------===//

def Executor_TruncfOp : Executor_FloatCastOp<"truncf"> {
  let description = [{
    Truncates a floating point type from one bitwidth to a floating point
    type of a narrower bitwidth. When an exact conversion is not possible,
    rounding occurs using the default rounding mode.
  }];
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def Executor_SIToFPOp : Executor_Op<"sitofp", [Pure]> {
  let description = [{
    `executor.sitofp` casts signed integer to the corresponding floating point
    value. This is similar to `static_cast` in C++. If value can't be exactly
    represented, it is rounded (implementation-defined).
  }];
  let arguments = (ins AnyTypeOf<Executor_ArithIntTypes>:$operand);
  let results = (outs AnyTypeOf<Executor_ArithFloatTypes>:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `to` type($result)";
}

def Executor_UIToFPOp : Executor_Op<"uitofp", [Pure]> {
  let description = [{
    `executor.uitofp` casts unsigned integer to the corresponding floating point
    value. If value can't be exactly represented, it is rounded
    (implementation-defined). Out-of-range casts (truncations) result in mapping
    to +/-inf or max/min value depending on whether the target floating point
    type has a representation for infinities.
  }];
  let arguments = (ins AnyTypeOf<Executor_ArithIntTypes>:$operand);
  let results = (outs AnyTypeOf<Executor_ArithFloatTypes>:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def Executor_FPToSIOp : Executor_CastOp<"fptosi"> {
  let description = [{
    `executor.fptosi` casts floating point value to the nearest signed integer
    value. This is similar to `static_cast` in C++.
  }];
  let arguments = (ins AnyTypeOf<Executor_ArithFloatTypes>:$operand);
  let results = (outs AnyTypeOf<Executor_ArithIntTypes>:$result);
}

//===----------------------------------------------------------------------===//
// TruncOp
//===----------------------------------------------------------------------===//

def Executor_TruncOp : Executor_CastOp<"trunc"> {
  let description = [{
    The `executor.trunc` op truncates an integer to a smaller bitwidth.
  }];
  let arguments = (ins Executor_Integer:$operand);
  let results =(outs Executor_Integer:$result);
}

//===----------------------------------------------------------------------===//
// ZExtOp
//===----------------------------------------------------------------------===//

def Executor_ZExtOp : Executor_CastOp<"zext"> {
  let description = [{
    `executor.zext` extends the bitwidth of an integer, filling in the
    new bits with zeros.
  }];
  let arguments = (ins Executor_Integer:$operand);
  let results =(outs Executor_Integer:$result);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SIExtOp
//===----------------------------------------------------------------------===//

def Executor_SIExtOp : Executor_CastOp<"siext"> {
  let description = [{
    `executor.siext` extends the bitwidth of an integer, by appending bits to
    the most significant side of the nubmer such that the positive/negative
    sign of the number is preserved. The positive/negative sign of the number
    is determined by interpreting the `operand` as an integer in two's complement
    format.
  }];
  let arguments = (ins Executor_Integer:$operand);
  let results =(outs Executor_Integer:$result);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SMinOp / SMaxOp / UMinOp / UMaxOp
//===----------------------------------------------------------------------===//

class Executor_MinMaxOp<string mnemonic, Type type> : Executor_Op<mnemonic, [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>,
    Commutative]> {
  let assemblyFormat = "operands attr-dict `:` type($lhs)";
  let arguments = (ins type:$lhs, type:$rhs);
  let results = (outs type:$result);
}

def Executor_SMaxOp : Executor_MinMaxOp<"smax", Executor_Integer>;
def Executor_SMinOp : Executor_MinMaxOp<"smin", Executor_Integer>;
def Executor_UMaxOp : Executor_MinMaxOp<"umax", Executor_Integer>;
def Executor_UMinOp : Executor_MinMaxOp<"umin", Executor_Integer>;
def Executor_FMaxOp : Executor_MinMaxOp<"fmax", Executor_Float>;
def Executor_FMinOp : Executor_MinMaxOp<"fmin", Executor_Float>;


//===----------------------------------------------------------------------===//
// Table Ops
//===----------------------------------------------------------------------===//

def Executor_CreateTableOp : Executor_Op<"table.create", [Pure]> {
  // Can optionally accept some number of initial values.
  let arguments = (ins Variadic<Executor_AnyType>:$init);
  let results = (outs Executor_Table:$result);
  let assemblyFormat = "attr-dict (`(` $init^ `:` type($init) `)`)? `:` type($result)";
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Type":$result), [{
      build($_builder, $_state, result, ValueRange{});
    }]>,
    // Convenience builders for constructing memref descriptors.
    OpBuilder<(ins "Type":$result, "Value":$allocatedPtr, "Value":$alignedPtr,
      "Value":$offset, "ValueRange":$sizes, "ValueRange":$strides)>,
    OpBuilder<(ins "Type":$result, "Value":$allocatedPtr, "Value":$alignedPtr,
      "OpFoldResult":$offset, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides)>
  ];
}

def Executor_InsertTableValueOp : Executor_Op<"table.set", [
  Pure, AllTypesMatch<["table", "result"]>
]> {
  let description = [{
    The `executor.table.insert` operation insert `value` into the `index`-th
    position in the table, returning a new (updated) copy of the table.
  }];

  let arguments = (ins Executor_Table:$table,
                       Executor_AnyType:$value,
                       I64Attr:$index);
  let results = (outs Executor_AnyType:$result);

  let assemblyFormat = [{
    attr-dict $value `into` $table `[` $index `]` `:` type($value) `,` type($table)
  }];

  let hasVerifier = 1;
}

def Executor_ExtractTableValueOp : Executor_Op<"table.get", [
  Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let description = [{
    The `executor.table.extract` operation returns the element at the `index-th`
    position in the given table.
  }];

  let arguments = (ins Executor_Table:$table,
                       I64Attr:$index);
  let results = (outs Executor_AnyType:$result);

  let assemblyFormat = [{
    attr-dict $table `[` $index `]` `:` type($table)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Executor_DynamicExtractTableValueOp : Executor_Op<"table.dynamic_get", [
  Pure]> {

  let description = [{
    The `executor.table.dynamic_get` operation returns the element at the
    `index-th` position in the given table. `index` is a variable.
  }];

  let arguments = (ins Executor_Table:$table,
                       Executor_Index:$index);
  let results = (outs Executor_AnyType:$result);

  let assemblyFormat = [{
    attr-dict $table `[` $index `]` `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def Executor_PrintOp : Executor_Op<"print", [
      MemoryEffects<[MemWrite]>]> {
  let description = [{
    The `executor.print` op prints a formatted string to "the output stream",
    the specifics of which depend on the runtime implementation.

    The operation takes an optional format string, which follows the C
    `sprintf`-style string formatting rules.
  }];
  let arguments = (ins Variadic<AnyType>:$arguments, OptionalAttr<StrAttr>:$format);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict ($format^)? `(` ($arguments^ `:` type($arguments))?  `)`
  }];
}

//===----------------------------------------------------------------------===//
// AlignToOp
//===----------------------------------------------------------------------===//

def Executor_AlignToOp : Executor_Op<"alignto", [Pure,
    AllTypesMatch<["arg", "result"]>,
    DeclareOpInterfaceMethods<RuntimeBuiltinInterface, ["lowerToCall"]>]> {

  let summary = "rounds up integer operand to nearest multiple of power-of-two alignment";

  let description = [{
    `executor.alignto` returns an integer formed from taking `arg` and
    rounding it up to the nearest multiple of `alignment`, which must be a power
    of two.

    One possible implementation of this is:

    ```
    bumped = arg + alignment -1
    result = bumped - urem(bumped, alignment)
    ```
  }];

  let arguments = (ins Executor_Index:$arg,
    UI32Attr:$alignment);

  let results = (outs Executor_Index:$result);
  let assemblyFormat = "attr-dict $arg `,` $alignment `:` type($arg)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetOffsetOp
//===----------------------------------------------------------------------===//

def Executor_GetOffsetOp : Executor_Op<"getoffset", [Pure]> {
  let summary = "calculates byte offsets to positions within aggregate types";

  let description = [{
    The `executor.getoffset` operation is used to calculate byte offsets within
    an aggregate type (e.g. array or table/struct). It is semantically equivalent
    to the [LLVM GetElementPtr (GEP) operation](https://llvm.org/docs/GetElementPtr.html),
    except that the `executor.getoffset` operation does not perform pointer arithmetic,
    it only calculates offsets. While the LLVM GEP op takes a pointer operand to
    which the calculated offset is added, this operation does not take a pointer
    operand and simply returns the offset which should be added to a base pointer
    assuming the base pointer points to data of `elem_type`.

    Since `executor` load/store operations take byte offsets as arguments,
    we omitted the pointer modification, especially since doing so would
    require materialization of excess `inttoptr|ptrtoint` ops.

    This operation may be further lowered when targeting interpreters; in
    such cases the `executor-expand-ops` pass can be invoked with
    `lower-getoffset=true` to lower `executor.getoffset` to more primitive
    ops.

    Note that internal storage of `staticIndices` uses I64, but if the
    DataModel specifies that the pointer or index type width is 32, then
    static indices should be convertible to i32 losslessly.
  }];

  let arguments =  (ins
    Variadic<Executor_Index>:$dynamicIndices,
    DenseI64ArrayAttr:$staticIndices,
    TypeAttr:$elem_type);

  let results = (outs Executor_Index:$result);
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Type":$elementType,
      "ArrayRef<OpFoldResult>":$indices)>,
    OpBuilder<(ins "Type":$resultType, "Type":$elementType,
      "ArrayRef<int64_t>":$indices)>
  ];

  let assemblyFormat = [{
    `[` custom<ExecutorMixedIndices>($dynamicIndices, $staticIndices) `]` attr-dict
    `:` functional-type(operands, results) `,` $elem_type
  }];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getIndices();
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Memory Ops
//===----------------------------------------------------------------------===//

def Executor_UndefinedOp : Executor_Op<"undef", [Pure]> {
  let summary = "produces an undefined value";

  let description = [{

    The `executor.undefined` operation produces a value whose binary
    storage representation (of the correct size for the scalar or fixed size
    aggregate) contains undefined data, and therefore may contain any possible
    bit pattern

    Current support for lowering `executor.undefined` is very limited. It
    is only created at intermediate stages (e.g. to support mem2reg transform,
    and translation to certain targets like the Lua interpreter will yield
    an error if the IR contains lingering `undef` values).

    See also [discussion on LLVM's undef](https://llvm.org/docs/LangRef.html#undefined-values)
    values for background on why this op exists and ideas for what we can
    do with it in the future.
  }];

  let results = (outs AnyTypeOf<[
    Executor_Integer, Executor_Float, Executor_Table, Executor_Ptr]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Executor_AllocaOp : Executor_Op<"alloca", [
      DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
      DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
]> {
  let summary = "block-scoped temporary allocation";

  let description = [{
    The `executor.alloca` operation allocates temporary memory to be
    automatically released when control transfers back from the region of its
    closest surrounding operation with an
    [`AutomaticAllocationScope`](../Traits.md/#automaticallocationscope) trait.

    The size of the allocation is specified by giving the element type and
    number of elements of an array which it must be large enough to hold.
    Once allocated, the size of the allocation cannot change.

    An optional alignment can be specified, the alignment is implicitly
    interpreted as the DataLayout-specified ABI alignment for the
    `element_type`.
  }];

  let arguments = (ins Executor_Index:$num_elements,
                       OptionalAttr<UI32Attr>:$alignment,
                       TypeAttr:$element_type);
  let results = (outs Res<Executor_HostPtr, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $num_elements `x` $element_type (`align` `(` $alignment^ `)`)?
      attr-dict `:` functional-type(operands, results)
  }];
}

def Executor_AllocateOp : Executor_Op<"alloc", []> {
  let description = [{
    `executor.alloc` performs an allocation of `num_bytes` using the given
    allocation. It returns an opaque pointer representing the allocation.
    The caller must also specify `alignment`.

    Backends/runtimes impleemnting this operation may automatically round up
    `num_bytes` to the next multiple of the `alignment`. `num_bytes` should
    be a power of two.
  }];
  let arguments = (ins Executor_Index:$num_bytes, Executor_Index:$alignment);
  let results = (outs Arg<Executor_HostPtr, "pointer to the allocated memory", [MemAlloc]>:$result);
  let assemblyFormat = [{
    attr-dict $num_bytes `bytes` `align` `(` $alignment `)` `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Executor_DeallocateOp : Executor_Op<"dealloc", []> {
  let arguments = (ins Arg<Executor_HostPtr, "pointer to the memory to be deallocated", [MemFree]>:$ptr);
  let assemblyFormat = [{
    attr-dict $ptr `:` type($ptr)
  }];
}

def Executor_LoadOp : Executor_Op<"load", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>]> {
  let description = [{
    `executor.load` loads `result` from the memory location given by `ptr +
    offset`. The offset is in terms of bytes.
  }];
  let arguments = (ins Arg<Executor_Ptr,"pointer to load from", [MemRead]>:$ptr,
                       Executor_Index:$offset);
  let results = (outs AnyTypeOf<[Executor_Integer, Executor_Float, Executor_Table, Executor_Ptr]>:$result);

  let assemblyFormat = [{
    attr-dict $ptr `+` $offset `:` functional-type(operands, results)
  }];
}

def Executor_StoreOp : Executor_Op<"store", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
  DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>]> {
  let description = [{
    `executor.store` stores `value` to the memory location given by `ptr +
    offset`. The offset is in terms of bytes.
  }];
  let arguments = (ins
    Arg<Executor_Ptr,"pointer to store to", [MemWrite]>:$ptr,
    Executor_Index:$offset,
    AnyTypeOf<[Executor_Ptr, Executor_Integer, Executor_Float, Executor_Table]>:$value);

  let assemblyFormat = [{
    attr-dict $value `to` $ptr `+` $offset
     `:` type($value) `,` qualified(type($ptr)) `,` type($offset)
  }];
}

def Executor_MemcpyOp : Executor_Op<"memcpy", [
  AllTypesMatch<["src_offset", "dest_offset", "num_bytes"]>
]> {
  let description = [{
    `executor.memcpy` copies `num_bytes` bytes from `source` to `dest`. Both
    buffers must be host memory buffers.
  }];
  let arguments = (ins Arg<Executor_HostVisiblePtr,"source pointer", [MemReadAt<0, FullEffect>]>:$src,
                       Executor_Index:$src_offset,
                       Arg<Executor_HostVisiblePtr,"destination pointer", [MemWriteAt<1, FullEffect>]>:$dest,
                       Executor_Index:$dest_offset,
                       Executor_Index:$num_bytes);
  let assemblyFormat = [{
    attr-dict $src `+` $src_offset `to` $dest `+` $dest_offset `size` $num_bytes
     `:` type(operands)
  }];
}

def Executor_StridedMemrefCopyOp : Executor_Op<"strided_memref_copy", []> {
  let description = [{
    Executes a copy from a source buffer to a destination buffer where the
    source and destination may be strided memrefs (with potentially different strides,
    but of the same shape).

    The `source_aligned_ptr` and `destination_aligned_ptr` are pointers to the aligned data
    of the source and destination memrefs, respectively while `src_descriptor` and `dest_descriptor`
    are pointers to the caller-allocated MLIR C-ABI compatible ranked memref
    descriptors provided for callee use.

    When this operation is lowered to the target, only the descriptor is passed. The
    data pointers are present here in order to capture the full memory effects of the
    operation.
  }];
  let arguments = (ins Arg<I32, "rank of source and destination memrefs">:$rank,
                       Arg<Executor_Index, "element size in bytes">:$elem_size,
                       Arg<Executor_HostPtr, "pointer to array of i64 values representing the shape of the source and destination memrefs",
                           [MemReadAt<0>]>:$shape,
                       // Source descriptor args
                       Arg<Executor_HostVisiblePtr,"source aligned data pointer", [MemReadAt<0>]>:$src_aligned_ptr,
                       Arg<Executor_Index, "source offset in elements to start of buffer">:$src_offset,
                       Arg<Executor_HostPtr,"source data strides in elements", [MemReadAt<0>]>:$src_strides,
                       // Destination descriptor args
                       Arg<Executor_HostVisiblePtr, "destination aligned data pointer", [MemWriteAt<1>]>:$dest_aligned_ptr,
                       Arg<Executor_Index, "destination offset in elements to start of buffer">:$dest_offset,
                       Arg<Executor_HostPtr, "destination data strides in elements", [MemReadAt<0>]>:$dest_strides);
  let assemblyFormat = [{
    attr-dict `(` operands `)`
     `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// Pointer<->Int Cast Ops
//===----------------------------------------------------------------------===//

class Executor_PointerCastOp <string mnemonic, list<Trait> traits = []> :
  Executor_Op<mnemonic,
      !listconcat(traits, [Pure])> {
  let assemblyFormat = [{
    attr-dict $arg `:` functional-type(operands, results)
  }];
}

def Executor_PtrToIntOp : Executor_PointerCastOp<"ptrtoint"> {
  let arguments = (ins Executor_Ptr:$arg);
  let results = (outs Executor_Index:$result);
}

def Executor_IntToPtrOp : Executor_PointerCastOp<"inttoptr"> {
  let arguments = (ins Executor_Index:$arg);
  let results = (outs Executor_Ptr:$result);
}


//===----------------------------------------------------------------------===//
// ABI Operations
//===----------------------------------------------------------------------===//

def Executor_ABIRecvOp : Executor_Op<"abi.recv", []> {
  let summary = "receive an argument from the caller";
  let description = [{
    The `executor.abi.recv` operation represents loading an argument value
    indirectly from a function input argument which has host pointer type.

    The argument must be annotated with an `#executor.arg<byval, ...>` argument
    ABI. Currently, only non-scalar value types are supported for passing
    inputs via pointer arguments.

    The result type of the operation must match the type of the argument's
    `#executor.arg<byval, ...>` value type.
  }];
  let arguments = (ins Arg<Executor_HostPtr,"address where value is read", [MemRead]>:$ptr,
       OptionalAttr<AnyAttr>:$memory_space);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Executor_ABISendOp : Executor_Op<"abi.send", [
  AllTypesMatch<["value", "result"]>
]> {
  let summary = "send an argument to the caller";
  let description = [{
    The `executor.abi.send` operation "stores" value to a function output argument
    which has host poitner type and is annotated with `#executor.arg<byref, ...>`
    argument ABI.

    The value type of the argument's `#executor.arg<byref, ...>` must match the type
    of `value`.
  }];
  let arguments = (ins AnyType:$value, Arg<Executor_HostPtr,"address where value is written", [MemWrite]>:$ptr, Optional<I1>:$ownership);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value `to` $ptr ( `ownership` `(` $ownership^ `)`)? attr-dict `:` type($value)";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$ptr), [{
      build($_builder, $_state, value.getType(), value, ptr, Value{});
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Runtime Plugin Operations
//===----------------------------------------------------------------------===//

def Executor_PluginOp : Executor_Op<"plugin", [Symbol]> {
  let summary = "declare a plugin handle";
  let description = [{
    The `executor.plugin` operation defines a new global symbol representing
    a handle to a plugin instance that can be invoked with
    `executor.call_plugin`.

    For TVM FFI backend (`ffi_backend = #executor.ffi_backend<tvm_ffi>`):
    This operation declares a callable handle to a TVM FFI function. The
    `plugin_name` identifies the TVM FFI library (e.g., a shared library path),
    and `function_name` identifies the function symbol within that library.

    During lowering, this operation is converted to a global variable that
    holds an opaque callable handle. The handle is initialized at module load
    time by calling the runtime function `_create_plugin_callable_tvm_ffi`,
    which:
    - Loads the TVM FFI library from `plugin_name` (caching loaded libraries)
    - Resolves the function symbol `function_name` from the library
    - Returns an opaque handle wrapping the `tvm::ffi::Function` object

    The `function_type` attribute specifies the MLIR function signature
    (arguments and results), which is used to validate `executor.call_plugin`
    calls. The `config` dictionary may contain additional backend-specific
    configuration attributes, but for the TVM FFI backend, it is currently
    unused.

    Example:

    ```mlir
    executor.plugin @my_plugin {
      plugin_name = "libflashinfer.so",
      function_name = "single_prefill_with_kv_cache",
      ffi_backend = #executor.ffi_backend<tvm_ffi>,
      function_type = (tensor<*xf32>, tensor<*xf32>) -> (),
      config = {}
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$plugin_name,
                       StrAttr:$function_name,
                       Executor_FFIBackendAttr:$ffi_backend,
                       OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
                       OptionalAttr<DictionaryAttr>:$config
                       );
  let results = (outs );
  let assemblyFormat = "$sym_name attr-dict `:` $function_type";

  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$symName,
      "llvm::StringRef":$pluginName, "llvm::StringRef":$functionName,
      "executor::FFIBackend":$ffiBackend,
      "FunctionType":$functionType,
      CArg<"DictionaryAttr", "{}">:$config)>
  ];

  let skipDefaultBuilders = 1;
}

def Executor_CallPluginOp : Executor_Op<"call_plugin", [
   AttrSizedOperandSegments,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call a plugin callable";
  let description = [{
    The `executor.call_plugin` operation calls an instance of a plugin
    callable.

    For TVM FFI backend (`ffi_backend = #executor.ffi_backend<tvm_ffi>`):
    This operation invokes a TVM FFI function following the TVM FFI ABI calling
    convention. All arguments and outputs are converted to `TVMFFIAny` structures
    and arranged according to the `arg_spec` attribute.

    Argument Specification (`arg_spec`):
    The `arg_spec` is a list of strings that encodes how to arrange arguments
    for the TVM FFI call. Each string can be:
    - `"args.N"`: Use the N-th input argument from `args`
    - `"rets.N"`: Use the N-th output argument from `outputs`
    - `"attrs.name"`: Use the attribute named `name` from `immediate_args`
    - `"none"`: Insert an optional None value

    Argument Conversion:
    - POD values (integers, floats): Stored directly in the 16-byte `TVMFFIAny`
      union structure without heap allocation
    - Tensor values: Converted to DLPack-compatible `DLTensor` structures and
      stored as pointers (`kTVMFFIDLTensorPtr`). The DLTensor contains:
      - Data pointer and device information (CUDA, CPU, etc.)
      - Rank, shape, and strides arrays (promoted to host memory)
      - Data type information (DLDataType)
      - Byte offset for non-contiguous views
    - Attributes: Converted to appropriate `TVMFFIAny` values (POD or string literals)

    Output Handling:
    TVM FFI functions do not return values through a result parameter. All
    outputs must be provided as input arguments with I/O aliasing specified
    via the `io_aliasing` attribute. This attribute maps each input index to
    an output index (-1 if not aliased), allowing in-place operations.

    Stream Handling:
    The optional `stream` operand specifies a CUDA stream for async operations.
    If not provided, NULL will be passed as the stream argument to the
    TVM FFI call.

    Memory Management:
    - DLTensor structures are allocated on the stack (via alloca) and contain
      pointers to the actual tensor data
    - Shape and strides arrays are promoted to host-allocated memory for runtime
      access
    - The `TVMFFIAny` argument array is allocated on the host stack for passing
      to the FFI runtime

    During lowering, this operation is converted to a call to the runtime
    function `_call_plugin_tvm_ffi`, which:
    1. Sets CUDA streams for DLTensor arguments (if applicable)
    2. Invokes the TVM FFI function using the `TVMFFISafeCallType` signature:
       `int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args,
                                  int32_t num_args, TVMFFIAny* result)`
    3. Handles errors and propagates them appropriately

    Example:

    ```mlir
    %result = executor.call_plugin @my_plugin
      ins(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>)
      outs(%output : tensor<*xf32>)
      {
        arg_spec = ["args.0", "args.1", "rets.0", "attrs.backend"],
        io_aliasing = [-1, -1, 0],
        immediate_args = {backend = "paged"}
      } : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> ()
    ```

    In this example:
    - `args.0` and `args.1` are the input tensors
    - `rets.0` is the output tensor (aliased with input 0 via `io_aliasing`)
    - `attrs.backend` is an immediate string attribute
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Optional<Executor_HostPtr>:$stream,
                       Variadic<AnyType>:$args,
                       Variadic<AnyType>:$outputs,
                       OptionalAttr<DictionaryAttr>:$immediate_args,
                       StrArrayAttr:$arg_spec,
                       OptionalAttr<DenseI32ArrayAttr>:$io_aliasing,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee
    (`stream` `(` $stream^ `)`)?
    `ins` `(` $args `:` type($args) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "::mlir::executor::PluginOp":$callee,
      "Value":$stream, "ValueRange":$args, "ValueRange":$outputs,
      "DictionaryAttr":$immediate_args,
      "::llvm::ArrayRef<llvm::StringRef>":$arg_spec,
      "::llvm::ArrayRef<int32_t>":$io_aliasing,
      CArg<"ArrayAttr", "{}">:$arg_attrs,
      CArg<"ArrayAttr", "{}">:$res_attrs)>,
    OpBuilder<(ins "::llvm::StringRef":$callee,
      "Value":$stream, "ValueRange":$args, "ValueRange":$outputs,
      "DictionaryAttr":$immediate_args,
      "ArrayAttr":$arg_spec,
      "DenseI32ArrayAttr":$io_aliasing,
      CArg<"ArrayAttr", "{}">:$arg_attrs,
      CArg<"ArrayAttr", "{}">:$res_attrs)>
  ];

  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// BufferBitcastOp
//===----------------------------------------------------------------------===//

def Executor_BufferBitcastOp : Executor_Op<"buffer_bitcast", [
      Pure,
      DeclareOpInterfaceMethods<ViewLikeOpInterface>,
      DeclareOpInterfaceMethods<CastOpInterface, ["areCastCompatible"]>]> {
  let description = [{
    `executor.buffer_bitcast` op performs a bitcast from one tensor or
    memref type to another tensor or memref type.

    The result element type must have the same bit width as the source
    element type.
  }];
  let arguments = (ins AnyShaped:$source);
  let results = (outs AnyShaped:$result);
  let assemblyFormat = "attr-dict $source `:` type($source) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// FileArtifactOp
//===----------------------------------------------------------------------===//

def Executor_FileArtifactOp : Executor_Op<"file_artifact", []> {
  let description = [{
    `executor.file_artifact` op declares a file artifact that will be
    emitted to the artifacts directory.
  }];
  let arguments = (ins StrAttr:$path, ElementsAttr:$data);
  let results = (outs );
  let assemblyFormat = "$path `data` `(` $data `)` attr-dict";
}

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOROPS_TD
