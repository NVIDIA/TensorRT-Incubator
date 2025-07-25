#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANDIALECT
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANDIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def Plan_Dialect : Dialect {
  let name = "plan";
  let summary = "dialect for program segmentation and parameter selection";
  let description = [{

    The Plan dialect consists of operations and transformations that assist with
    the segmentation of an input program into logical groups of
    operations that represent operations which should be compiled in different
    pipelines.

  }];
  let cppNamespace = "::mlir::plan";

  let hasRegionArgAttrVerify = 1;

  let extraClassDeclaration = [{

    /// The attribute name that is associated with a SymbolRefAttr for `func.func`
    /// operations that have an associated function which computes result shapes from
    /// input shapes.
    static constexpr llvm::StringLiteral kShapeFuncAttrName = "plan.shape_func";

    /// The attribute name indicates that the `func.func` is a shape func. This marker
    /// is used to skip bounds analysis for shape funcs.
    static constexpr llvm::StringLiteral kShapeFuncMarkerAttrName = "plan.shapes_func_marker";

    /// Attribute attached to functions which specifies what type of cluster the outlined
    /// function originated from.
    static constexpr llvm::StringLiteral kFuncTargetKind = "plan.cluster_kind";

    /// Attribute attached to top-level modules which dictate which cluster kinds
    /// should be used to drive compilation.
    static constexpr llvm::StringLiteral kModuleClusterKindsAttrName = "plan.cluster_kinds";

    /// The name of the function argument attribute that indicates an argument
    /// is tied with a result. Arguments tagged in this manner are associated
    /// with results based on the order in the argument list.
    static constexpr StringRef kResultArgAttrName = "plan.result_arg";

    /// List of known Compilation Task Extension constructors.
    mlirtrt::compiler::ExtensionConstructorRegistry extensionConstructors;

    template<typename, typename ...>
    friend class PlanDialectExtension;

    /// Parses an attribute registered by this dialect or one of its extensions.
    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

    /// Prints an attribute registered by this dialect or one of its extensions.
    void printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter & printer)
        const override;

    using AttrPrintingHook = std::function<void(Attribute, DialectAsmPrinter&)>;
    using AttrParsingHook = std::function<Attribute(DialectAsmParser &, Type)>;

    template<typename Attr>
    void addExtensionAttribute();

    template<typename Op>
    void addExtensionOperation();

    // Registers types specified as template parameters with the Plan
    /// dialect.
    template <typename... Attributes>
    void addAttributesExt() {
      (addExtensionAttribute<Attributes>(), ...);
    }

    // Registers types specified as template parameters with the Plan
    /// dialect.
    template <typename... Ops>
    void addExtensionOperations() {
      (addExtensionOperation<Ops>(), ...);
    }

    /// Return the name of the function arg/result attributes that encode
    /// host tensor value bounds. It should have a type `plan::BoundsAttr`.
    static StringRef getValueBoundsAttrName() {
      return "plan.value_bounds";
    }

    /// Return the name of the function arg/result attributes that encode
    /// the shape bounds. It should have a type `plan::BoundsAttr`.
    static StringRef getShapeBoundsAttrName() {
      return "plan.shape_profile";
    }

    /// Return the name of the attribute used to encode memory space
    /// constraints. It should appear in function attributes or in
    /// function arg/result attribute dictionaries.
    static StringRef getMemorySpaceConstraintAttrName() {
      return "plan.memory_space";
    }


  private:
    ::llvm::StringMap<AttrParsingHook> attrParsingHooks;
    ::llvm::DenseMap<::mlir::TypeID, AttrPrintingHook> attrPrintingHooks;
  }];
}


class Plan_I32EnumAttr<string name, string description,
  list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::plan";
  let genSpecializedAttr = 0;
}

class Plan_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic, list<Trait> traits = []> :
  EnumAttr<Plan_Dialect, intEnumAttr, mnemonic, traits> {
  let assemblyFormat = "`<` $value `>`";
}

class Plan_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Plan_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANDIALECT
