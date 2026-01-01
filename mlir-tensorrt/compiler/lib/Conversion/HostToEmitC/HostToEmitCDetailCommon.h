//===- HostToEmitCDetailCommon.h --------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Internal helpers for the HostToEmitC conversion.
///
//===----------------------------------------------------------------------===//
#ifndef CONVERSION_HOSTTOEMITC_HOSTTOEMITCDETAILCOMMON
#define CONVERSION_HOSTTOEMITC_HOSTTOEMITCDETAILCOMMON

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::host_to_emitc {

inline static FunctionOpInterface getParentFunction(Operation *op) {
  assert(op && "expected operation");
  auto funcLike = dyn_cast<FunctionOpInterface>(op);
  return funcLike ? funcLike : op->getParentOfType<FunctionOpInterface>();
}

/// Convenience builder for `emitc::CallOpaqueOp` that accepts a mixed list of
/// SSA operands (`Value`) and immediate arguments (`Attribute`) encoded as
/// `OpFoldResult`s.
template <typename ArgsType = ArrayRef<OpFoldResult>>
inline static emitc::CallOpaqueOp
createCallOpaque(OpBuilder &b, Location loc, TypeRange resultTypes,
                 StringRef callee, ArgsType callArgs,
                 ArrayRef<Attribute> templateArgs = {}) {
  SmallVector<Value> operands;
  SmallVector<Attribute> args;
  operands.reserve(callArgs.size());
  args.reserve(callArgs.size());

  using ValueType = llvm::remove_cvref_t<decltype(callArgs.front())>;

  static_assert(std::is_same_v<ValueType, OpFoldResult> ||
                    std::is_same_v<ValueType, Value>,
                "Unsupported ArgsType");

  if constexpr (std::is_same_v<ValueType, OpFoldResult>) {
    for (OpFoldResult arg : callArgs) {
      if (Value v = arg.dyn_cast<Value>()) {
        unsigned idx = operands.size();
        operands.push_back(v);
        args.push_back(b.getIndexAttr(idx));
        continue;
      }
      args.push_back(llvm::cast<Attribute>(arg));
    }
  }

  if constexpr (std::is_same_v<ValueType, Value>) {
    for (Value arg : callArgs) {
      unsigned idx = operands.size();
      operands.push_back(arg);
      args.push_back(b.getIndexAttr(idx));
    }
  }

  ArrayAttr argsAttr = args.empty() ? ArrayAttr() : b.getArrayAttr(args);
  ArrayAttr templateArgsAttr =
      templateArgs.empty() ? ArrayAttr() : b.getArrayAttr(templateArgs);

  return emitc::CallOpaqueOp::create(b, loc, resultTypes, callee, argsAttr,
                                     templateArgsAttr, operands);
}

inline static emitc::FuncOp insertEmitCFunction(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name,
    Type resultType, TypeRange args,
    std::function<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(module.getBody());
  auto func = emitc::FuncOp::create(
      b, loc, name,
      FunctionType::get(b.getContext(), args,
                        resultType ? resultType : TypeRange{}));
  if (bodyBuilder) {
    Block *body = func.addEntryBlock();
    OpBuilder::InsertionGuard inner(b);
    b.setInsertionPointToStart(body);
    Value v = bodyBuilder(b, loc, body->getArguments());
    emitc::ReturnOp::create(b, loc, v);
  }
  return func;
}

inline static void emitStatusCheckOrAbort(OpBuilder &b, Location loc,
                                          Value status) {
  Operation *parentOp = b.getInsertionBlock()->getParentOp();
  assert(parentOp && "expected parent operation");
  if (auto funcLike = getParentFunction(parentOp)) {
    auto type = dyn_cast<FunctionType>(funcLike.getFunctionType());
    if (type && type.getNumResults() == 1 && type.getResult(0).isInteger(32)) {
      emitc::VerbatimOp::create(b, loc, "if ({} != 0) {{ return {}; }",
                                ValueRange{status, status});
      return;
    }
  }
  emitc::VerbatimOp::create(b, loc, "mtrt::abort_on_error({});", status);
}

} // namespace mlir::host_to_emitc

#endif // CONVERSION_HOSTTOEMITC_HOSTTOEMITCDETAILCOMMON
