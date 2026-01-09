#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_DIALECT
#define MLIR_TENSORRT_DIALECT_PLAN_IR_DIALECT

#include "mlir/IR/Dialect.h"

#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsDialect.h.inc"

template <typename Attr>
void mlir::plan::PlanDialect::addExtensionAttribute() {
  StringRef mnemonic = Attr::getMnemonic();
  attrParsingHooks.try_emplace(mnemonic, Attr::parse);
  attrPrintingHooks.try_emplace(TypeID::get<Attr>(),
                                [](Attribute attr, AsmPrinter &printer) {
                                  printer << cast<Attr>(attr).getMnemonic();
                                  cast<Attr>(attr).print(printer);
                                });
  addAttributes<Attr>();
}

template <typename Op>
void mlir::plan::PlanDialect::addExtensionOperation() {
  StringRef name = Op::getOperationName();
  MLIRContext *ctx = getContext();
  if (std::optional<RegisteredOperationName> regName =
          RegisteredOperationName::lookup(name, ctx)) {
    if (regName->getTypeID() != TypeID::get<Op>()) {
      llvm::report_fatal_error("PlanDialect attempted to register extension "
                               "duplicate operation with differing TypeIDs");
    }
    return;
  }
  addOperations<Op>();
}

namespace mlir::plan {

//===----------------------------------------------------------------------===//
// PlanDialectExtension
//===----------------------------------------------------------------------===//

/// PlanDialectExtension is the bae class for DialectExtensions to add
/// Attributes and Pipeline TaskExtensions into the PlanDialect.
template <typename DerivedTy, typename... ExtraDialects>
class PlanDialectExtension
    : public DialectExtension<DerivedTy, PlanDialect, ExtraDialects...> {

  using Initializer = std::function<void(PlanDialect *)>;
  using DialectLoader = std::function<void(MLIRContext *)>;

public:
  using Base = PlanDialectExtension<DerivedTy, ExtraDialects...>;

  StringRef getName() const { return ""; }

  /// Extension constructor.
  explicit PlanDialectExtension() { static_cast<DerivedTy *>(this)->init(); }

  /// Extension application hook. Actually loads the dependent dialects and
  /// registers the additional operations. Not expected to be called directly.
  void apply(MLIRContext *context, PlanDialect *planDialect,
             ExtraDialects *...) const final {
    for (const DialectLoader &loader : dialectLoaders)
      loader(context);
    for (const Initializer &init : initializers)
      init(planDialect);
  }

  /// Hook for derived classes to inject constructor behavior.
  void init() {}

  template <typename... AttrTypes>
  void registerAttributes() {
    initializers.push_back([](PlanDialect *planDialect) {
      planDialect->addExtensionAttribute<AttrTypes...>();
    });
  }

  template <typename... OpTypes>
  void registerOps() {
    initializers.push_back([](PlanDialect *planDialect) {
      planDialect->addExtensionOperations<OpTypes...>();
    });
  }

  template <typename... DialectTypes>
  void declareGeneratedDialects() {
    dialectLoaders.push_back(
        [](MLIRContext *ctx) { (ctx->getOrLoadDialect<DialectTypes>(), ...); });
  }

protected:
  /// Callbacks performing extension initialization.
  SmallVector<Initializer> initializers;

  /// Callbacks loading the dependent dialects.
  SmallVector<DialectLoader> dialectLoaders;
};
} // namespace mlir::plan

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_DIALECT
