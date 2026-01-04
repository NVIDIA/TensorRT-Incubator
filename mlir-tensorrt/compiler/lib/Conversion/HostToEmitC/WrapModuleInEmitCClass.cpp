
//===- WrapModuleInEmitCClass.cpp -----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Post-processing pass that wraps module-scope EmitC globals and entrypoints
/// into a single `emitc.class` state object.
///
/// The output of `convert-host-to-emitc` can contain:
///   - `emitc.global` declarations (module scope)
///   - helper functions `<name>_initialize*` / `<name>_destroy*`
///
/// Some embedding scenarios prefer a single C++ "program object" that owns all
/// globals as fields. This pass rewrites the module to the following intended
/// C++ shape (schematic):
///
///   struct <ModuleName>Program {
///     // fields corresponding to former `emitc.global`s
///     ...
///     void initialize(); // optionally inlines helper bodies
///     void destroy();
///     // former free functions become methods
///   };
//===----------------------------------------------------------------------===//
#include "HostToEmitCDetailCommon.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir::host_to_emitc;
namespace mlir {
#define GEN_PASS_DEF_WRAPMODULEINEMITCCLASSPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static std::string sanitizeCppIdentifier(llvm::StringRef s) {
  std::string out;
  out.reserve(s.size());
  auto isValidFirst = [](char c) {
    return llvm::isAlpha(static_cast<unsigned char>(c)) || c == '_';
  };
  auto isValid = [](char c) {
    return llvm::isAlnum(static_cast<unsigned char>(c)) || c == '_';
  };
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (i == 0 && !isValidFirst(c))
      out.push_back('_');
    out.push_back(isValid(c) ? c : '_');
  }
  if (out.empty())
    out = "Program";
  return out;
}

/// Wrap module-scope EmitC globals/functions into a single `emitc.class`
/// program wrapper and replace `emitc.get_global` uses with `emitc.get_field`.
static LogicalResult wrapEmitCInProgramClass(ModuleOp moduleOp,
                                             IRRewriter &rewriter) {
  MLIRContext *ctx = moduleOp->getContext();
  Location loc = moduleOp.getLoc();

  // EmitC upstream currently does not reliably support `emitc.get_field`
  // returning `!emitc.lvalue<T>` for non-array primitive/opaque types.
  // Workaround: store scalar fields as a size-1 array `!emitc.array<1xT>` and
  // access it via `get_field` + `subscript [0]` to obtain `!emitc.lvalue<T>`.
  //
  // Intended C++ for a scalar field `T x;` becomes:
  //   T x[1];
  // and later uses do:
  //   x[0] = ...;
  auto sizeTType = emitc::SizeTType::get(ctx);
  auto getIndex0 = [&](OpBuilder &b, Location l) -> Value {
    return b.create<emitc::ConstantOp>(l, sizeTType, b.getIndexAttr(0));
  };

  // Collect module-scope globals and functions to migrate.
  SmallVector<emitc::GlobalOp> globals;
  SmallVector<FunctionOpInterface> funcs;
  Block &moduleBlock = *moduleOp.getBody();
  for (Operation &op : moduleBlock) {
    if (auto g = dyn_cast<emitc::GlobalOp>(op))
      globals.push_back(g);
    else if (auto f = dyn_cast<FunctionOpInterface>(op))
      funcs.push_back(f);
  }

  // Only introduce a stateful Program class when the module actually has
  // stateful globals to manage.
  if (globals.empty())
    return success();

  SymbolTableCollection symbolTables;
  SymbolUserMap userMap(symbolTables, moduleOp);

  const std::string moduleName =
      (moduleOp.getSymName() ? *moduleOp.getSymName() : "unnamed_module").str();
  const std::string className = sanitizeCppIdentifier(moduleName) + "Program";

  // Create the class wrapper.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto classOp = rewriter.create<emitc::ClassOp>(
      loc, rewriter.getStringAttr(className), /*final_specifier=*/nullptr);
  Region &classRegion = classOp.getBody();
  if (classRegion.empty())
    classRegion.push_back(new Block());
  Block &classBlock = classRegion.front();

  // 1) Turn each module-scope `emitc.global` into a class `emitc.field`.
  struct FieldInfo {
    emitc::FieldOp field;
    Type originalType;    // Type of the original emitc.global
    Type fieldType;       // Type stored in the field
    bool isScalarWrapped; // Whether the field is `array<1xT>` wrapper
  };
  llvm::DenseMap<StringAttr, FieldInfo> fieldByGlobal;

  rewriter.setInsertionPointToStart(&classBlock);
  for (emitc::GlobalOp g : globals) {
    Type originalType = g.getType();

    // - For scalar/opaque/pointer types, store as `array<1xT>` so that we can
    //   take an lvalue via `subscript[0]`.
    // - If the global is already an array, store it directly as the field type
    //   (do NOT wrap; nested arrays are illegal in EmitC).
    bool isArray = isa<emitc::ArrayType>(originalType);
    Type fieldType = originalType;
    bool isScalarWrapped = false;
    if (!isArray) {
      fieldType = emitc::ArrayType::get({1}, originalType);
      isScalarWrapped = true;
    }

    // Preserve initializers for array-typed globals (e.g. memref.global
    // materializations) so conversion tests remain value-stable.
    Attribute initValue = isArray ? g.getInitialValueAttr() : Attribute{};

    auto field = rewriter.create<emitc::FieldOp>(g.getLoc(), g.getSymNameAttr(),
                                                 TypeAttr::get(fieldType),
                                                 /*initial_value=*/initValue);

    fieldByGlobal.try_emplace(
        g.getSymNameAttr(),
        FieldInfo{field, originalType, fieldType, isScalarWrapped});
  }

  // 2) Move module-scope EmitC functions into the class as methods.
  rewriter.setInsertionPointToEnd(&classBlock);
  for (FunctionOpInterface f : funcs)
    f->moveBefore(&classBlock, classBlock.end());

  // 3) Rewrite `emitc.get_global @X` to `emitc.get_field @X` inside the class.
  classOp.walk([&](emitc::GetGlobalOp op) {
    StringAttr key = StringAttr::get(ctx, op.getNameAttr().getValue());
    auto it = fieldByGlobal.find(key);
    if (it == fieldByGlobal.end())
      return;

    OpBuilder b(op);

    // If the original global was an array, `get_field` already yields the
    // correct type and we do not need to create a subscript lvalue.
    if (isa<emitc::ArrayType>(op.getType())) {
      auto getField = b.create<emitc::GetFieldOp>(
          op.getLoc(), it->second.fieldType,
          FlatSymbolRefAttr::get(ctx, key.getValue()));
      rewriter.replaceOp(op, getField.getResult());
      return;
    }

    // Scalar case: `get_field` returns `array<1xT>`, then `subscript[0]`
    // yields the lvalue to the element.
    auto getField = b.create<emitc::GetFieldOp>(
        op.getLoc(), it->second.fieldType,
        FlatSymbolRefAttr::get(ctx, key.getValue()));
    Value idx0 = getIndex0(b, op.getLoc());
    auto sub = b.create<emitc::SubscriptOp>(
        op.getLoc(), emitc::LValueType::get(it->second.originalType),
        getField.getResult(), ValueRange{idx0});
    rewriter.replaceOp(op, sub.getResult());
  });

  // 4) Create consolidated `initialize()` / `destroy()` methods and inline the
  // bodies of existing per-resource init/destroy functions.
  Type i32Type = rewriter.getI32Type();
  auto mkEmptyMethod = [&](StringRef name) -> emitc::FuncOp {
    OpBuilder b(ctx);
    b.setInsertionPointToEnd(&classBlock);
    auto fnType =
        b.getFunctionType(/*inputs=*/TypeRange{}, /*results=*/{i32Type});
    auto method = b.create<emitc::FuncOp>(loc, name, fnType);
    Block *entry = method.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());
    Value zero = emitc::ConstantOp::create(bodyBuilder, loc, i32Type,
                                           bodyBuilder.getZeroAttr(i32Type));
    bodyBuilder.create<emitc::ReturnOp>(loc, zero);
    return method;
  };

  emitc::FuncOp initAll = mkEmptyMethod("initialize");
  emitc::FuncOp destroyAll = mkEmptyMethod("destroy");

  SmallVector<FunctionOpInterface> initHelpers;
  SmallVector<FunctionOpInterface> destroyHelpers;
  for (auto f : classOp.getOps<FunctionOpInterface>()) {
    if (f == initAll || f == destroyAll || f.getNumArguments() != 0)
      continue;
    // We can only handle initialization functions that either return an error
    // code or no result.
    if (!f.getResultTypes().empty() && f.getResultTypes().front() != i32Type)
      return emitError(f.getLoc()) << "initialization function must either "
                                      "return an error code or no result";
    StringRef n = f.getName();
    if (n.ends_with("_initialize"))
      initHelpers.push_back(f);
    else if (n.ends_with("_destroy"))
      destroyHelpers.push_back(f);
  }

  auto inlineInto = [&](FunctionOpInterface src, emitc::FuncOp dst) {
    OpBuilder::InsertionGuard g(rewriter);
    Block &srcBlock = src.getFunctionBody().front();
    Block &dstBlock = dst.getFunctionBody().front();
    Operation *srcTerm = srcBlock.getTerminator();
    Operation *dstTerm = dstBlock.getTerminator();
    rewriter.inlineBlockBefore(&srcBlock, dstTerm);
    if (ValueRange returnedValues = srcTerm->getOperands();
        returnedValues.size() == 1) {
      assert(returnedValues.front().getType() == i32Type &&
             "expected error code return type");
      rewriter.setInsertionPoint(srcTerm);
      emitStatusCheckOrAbort(rewriter, src.getLoc(), returnedValues.front());
    }
    rewriter.eraseOp(srcTerm);
  };

  // Ensure the CUDA stream field is initialized to nullptr if present.
  if (auto it = fieldByGlobal.find(
          rewriter.getStringAttr(moduleName + "_cuda_stream"));
      it != fieldByGlobal.end()) {
    Block &dstBlock = initAll.getBody().front();
    Operation *dstTerm = dstBlock.getTerminator();
    OpBuilder b(dstTerm);

    // Stream is stored as array<1xCUstream>.
    auto fieldArr = b.create<emitc::GetFieldOp>(
        loc, it->second.fieldType,
        FlatSymbolRefAttr::get(ctx, it->second.field.getSymName()));
    Value idx0 = getIndex0(b, loc);
    auto elemLVal = b.create<emitc::SubscriptOp>(
        loc, emitc::LValueType::get(it->second.originalType), fieldArr,
        ValueRange{idx0});
    Value nullptrVal = b.create<emitc::ConstantOp>(
        loc, it->second.originalType, emitc::OpaqueAttr::get(ctx, "nullptr"));
    b.create<emitc::AssignOp>(loc, elemLVal, nullptrVal);
  }

  for (FunctionOpInterface f : initHelpers) {
    inlineInto(f, initAll);
  }
  for (FunctionOpInterface f : llvm::reverse(destroyHelpers))
    inlineInto(f, destroyAll);
  for (FunctionOpInterface f : initHelpers)
    rewriter.eraseOp(f);
  for (FunctionOpInterface f : destroyHelpers)
    rewriter.eraseOp(f);

  {
    SymbolTableCollection finalSymbolTables;
    SymbolUserMap finalUserMap(finalSymbolTables, moduleOp);
    // Erase module-scope globals only when they no longer have any remaining
    // `emitc.get_global` users.
    for (emitc::GlobalOp g : globals) {
      if (finalUserMap.getUsers(g).empty())
        rewriter.eraseOp(g);
    }
  }

  return success();
}

namespace {

struct WrapModuleInEmitCClassPass
    : public mlir::impl::WrapModuleInEmitCClassPassBase<
          WrapModuleInEmitCClassPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(moduleOp.getContext());
    if (failed(wrapEmitCInProgramClass(moduleOp, rewriter)))
      return signalPassFailure();
  }
};

} // namespace
