#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORDIALECT_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Executor Dialect
//===----------------------------------------------------------------------===//

def Executor_Dialect : Dialect {
  let name = "executor";
  let summary = "A low-level retargetable IR";
  let description = [{
    The Executor dialect is a low-level dialect that provides an IR whose
    abstraction level is between that of the upstream "standard" dialects (e.g.
    Arith, MemRef) and the upstream "LLVM" dialect.

    The purpose of the Executor dialect is to provide a constrained IR that we
    can use to represent host-side code. For example, we need host program
    representations in the following areas:

    - In runner executables, where it is currently very constraining to limit
      our executable representation to a single TensorRT engine.
    - To represent TensorRT plugins, which may need arbitrary setup/teardown
      host code as well as host code that performs kernel launches in the
      enqueue method.
    - For testing purposes, where we would like to describe programs in MLIR
      that interact with TensorRT's API to construct and build engines.

    Executor IR can be lowered in one of the following manners:

    - It could be interpreted directly for simple programs
    - It could be translated to scripting language or compiled to
      bytecode for a VM.
    - It could be further lowered to a dialect such as LLVM and compiled
      into host code.

    We will provide a reference translation to the Lua scripting language, since
    Lua is both extensible from C/C++ and the compiler/runtime VM is embeddable
    in any of our tools or integration binaries. The Lua target could refer to
    Lua textual form or the bytecode, which requires no further compilation to
    be executed at runtime.

    Toward these ends, the dialect provides a set of low-level instructions for
    basic arithmetic, but we allow for mixed-use of some ops in `func`, `scf`, and
    `cf` dialects. Our set of arithmetic is limited relative to `arith`. For
    example, we exlcude operations equivalent to `arith.udiv` from our supported
    set, although we could gradually widden the operator set as required.

    The Executor implicitly defines a notion of a "module" that can be loaded
    and executed. A module can contain a list of functions (which will be
    executed on the host). The module may reference/call outside functions or
    globals that are declared at the top of the module. We currently reused
    `builtin.module` for this representation. The mechanism/ABI for accessing
    these external resources are implementation defined.
  }];
  let cppNamespace = "::mlir::executor";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let hasRegionArgAttrVerify = 1;
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    Operation *materializeConstant(::mlir::OpBuilder &builder,
                                   ::mlir::Attribute value,
                                   ::mlir::Type type,
                                   ::mlir::Location loc) override;

    /// The function attribute name that carries the executor attribute
    /// containing logical information about the (packed) signature.
    static constexpr StringRef kFunctionMetadataAttrName
      = "executor.function_metadata";

    /// The function attribute name that specifies another function symbol
    /// name as being the associated "shape function" of the current function.
    static constexpr StringRef kShapeFuncAttrName = "executor.shape_func";

    /// The name of the function argument attribute that indicates an argument
    /// is tied with a result. Arguments tagged in this manner are associated
    /// with results based on the order in the argument list.
    static constexpr StringRef kResultArgAttrName = "executor.result_slot";

    /// The name of the function argument attribute that indicates an argument
    /// donated for result. Arguments tagged in this manner are associated
    /// with result at index represented by attribute value.
    static constexpr StringRef kDonatedArgAttrName = "executor.donated_arg";

    /// Name of the attribute attached to a module that describes the shape of the
    /// process grid for which the exported functions assume to be a part of.
    static constexpr StringRef kProcessGridShapeAttrName = "executor.process_grid_shape";

    /// Return the name of the function arg attr that encodes
    /// host tensor value bounds. It should have a type `executor::ValueBoundsAttr`.
    static StringRef getValueBoundsAttrName() {
      return "executor.value_bounds";
    }

    /// Return the name of the function arg attr that encodes
    /// the shape bounds. It should have a type `executor::DimensionBoundsAttr`.
    static StringRef getShapeBoundsAttrName() {
      return "executor.shape_profile";
    }

  private:
    /// Register attributes for the Executor dialect.
    void registerAttributes();
  }];
  // Temporary until CUDA ops are completely removed out of Executor dialect
  let dependentDialects = [
    "::mlir::DLTIDialect"
  ];
}

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTORDIALECT_TD
