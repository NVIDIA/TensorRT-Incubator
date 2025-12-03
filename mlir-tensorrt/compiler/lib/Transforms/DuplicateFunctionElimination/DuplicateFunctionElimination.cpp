//===- DuplicateFunctionElimination.cpp -----------------------------------===//
//
// This pass is modified from the upstream pass from upstream MLIR at
// llvm-project/mlir/lib/Dialect/Func/Transforms/DuplicateFunctionElimination.cpp
// which has LLVM license (Apache License v2.0 with LLVM Exceptions)
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// Extension of upstream duplicate-function-elimination pass except that
/// certain bugs are fixed and capability is slightly more general.
/// TODO: move these fixes upstream
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mtrt {
#define GEN_PASS_DEF_FUNCEXTDUPLICATEFUNCTIONELIMINATIONPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;

namespace {
// Define a notion of function equivalence that allows for reuse. Ignore the
// symbol name for this purpose.
struct DuplicateFuncOpEquivalenceInfo
    : public llvm::DenseMapInfo<func::FuncOp> {

  static unsigned getHashValue(const func::FuncOp cFunc) {
    if (!cFunc) {
      return DenseMapInfo<func::FuncOp>::getHashValue(cFunc);
    }

    // Aggregate attributes, ignoring the symbol name.
    llvm::hash_code hash = {};
    func::FuncOp func = const_cast<func::FuncOp &>(cFunc);
    StringAttr symNameAttrName = func.getSymNameAttrName();
    for (NamedAttribute namedAttr : cFunc->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      if (attrName == symNameAttrName)
        continue;
      hash = llvm::hash_combine(hash, namedAttr);
    }

    // Also hash the func body.
    func.getBody().walk([&](Operation *op) {
      hash = llvm::hash_combine(
          hash, OperationEquivalence::computeHash(
                    op, /*hashOperands=*/OperationEquivalence::ignoreHashValue,
                    /*hashResults=*/OperationEquivalence::ignoreHashValue,
                    OperationEquivalence::IgnoreLocations));
    });

    return hash;
  }

  static bool isEqual(func::FuncOp lhs, func::FuncOp rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Check discardable attributes equivalence
    if (lhs->getDiscardableAttrDictionary() !=
        rhs->getDiscardableAttrDictionary())
      return false;

    // Check properties equivalence, ignoring the symbol name.
    // Make a copy, so that we can erase the symbol name and perform the
    // comparison.
    auto pLhs = lhs.getProperties();
    auto pRhs = rhs.getProperties();
    pLhs.sym_name = nullptr;
    pRhs.sym_name = nullptr;
    if (pLhs != pRhs)
      return false;

    // Compare inner workings.
    return OperationEquivalence::isRegionEquivalentTo(
        &lhs.getBody(), &rhs.getBody(), OperationEquivalence::IgnoreLocations);
  }
};

class FuncExtDuplicateFuncEliminationPass
    : public mtrt::impl::FuncExtDuplicateFunctionEliminationPassBase<
          FuncExtDuplicateFuncEliminationPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    auto module = getOperation();
    SymbolTableCollection symbolTable;
    SymbolUserMap userMap(symbolTable, module);
    DenseSet<func::FuncOp> toBeErased;

    SymbolTable::walkSymbolTables(
        module, true,
        [&userMap, &toBeErased](Operation *symbolTable,
                                bool allSymUsesVisible) {
          if (symbolTable->getNumRegions() != 1)
            return;
          llvm::DenseSet<func::FuncOp, DuplicateFuncOpEquivalenceInfo>
              uniqueFuncOps;
          DenseMap<StringAttr, func::FuncOp> leaders;
          for (auto f : symbolTable->getRegion(0).getOps<func::FuncOp>()) {
            if (f.isDeclaration())
              continue;
            auto [repr, inserted] = uniqueFuncOps.insert(f);
            leaders[f.getSymNameAttr()] = *repr;
            if (!inserted) {
              toBeErased.insert(f);
              userMap.replaceAllUsesWith(f, repr->getSymNameAttr());
            }
          }
        });

    // Enumerate functions by enumerating the call functions.
    IRRewriter rewriter(&getContext());
    for (func::FuncOp it : toBeErased) {
      assert(SymbolTable::symbolKnownUseEmpty(it, getOperation()) &&
             "expected no uses");
      rewriter.eraseOp(it);
    }
  }
};
} // namespace
