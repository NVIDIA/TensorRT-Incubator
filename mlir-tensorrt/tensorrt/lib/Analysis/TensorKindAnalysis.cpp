//===- TensorKindAnalysis.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "tensor-kind-analysis"
#define DBGS() llvm::dbgs() << "[" << DEBUG_TYPE << "] "

using namespace mlir;

/// Returns `true` if the type is a scalar or a tensor with the number of
/// elements below the threshold.
static bool isScalarOrSmallTensor(Type v) {
  if (v.isIntOrIndexOrFloat())
    return true;
  return detail::isHostTensorCandidate(v);
}

static std::optional<Block *> getEntryBlock(FunctionOpInterface iface) {
  if (!iface.getFunctionBody().hasOneBlock())
    return {};
  return &iface.getFunctionBody().front();
}

static bool isFunctionArgument(Value operand) {
  auto blockArg = dyn_cast<BlockArgument>(operand);
  if (!blockArg)
    return false;
  auto func =
      blockArg.getParentRegion()->getParentOfType<FunctionOpInterface>();
  if (!func)
    return false;
  std::optional<Block *> entryBlock = getEntryBlock(func);
  return entryBlock && *entryBlock == blockArg.getOwner();
}

static bool isReturnedValue(OpOperand &operand) {
  if (!operand.getOwner()->hasTrait<OpTrait::ReturnLike>())
    return false;
  auto func = operand.getOwner()->getParentOfType<FunctionOpInterface>();
  if (!func)
    return false;
  std::optional<Block *> entryBlock = getEntryBlock(func);
  return entryBlock && *entryBlock == operand.getOwner()->getBlock() &&
         (*entryBlock)->getTerminator() == operand.getOwner();
}

static TensorKindInfo getFunctionArgInfo(Value value) {
  assert(isFunctionArgument(value));
  BlockArgument blockArg = cast<BlockArgument>(value);
  auto func =
      blockArg.getParentRegion()->getParentOfType<FunctionOpInterface>();
  TensorKindInfo argConstraint = TensorKindInfo(TensorKind::Device);
  if (func == blockArg.getParentBlock()->getParentOp() &&
      func.getArgAttr(blockArg.getArgNumber(), getHostTensorArgAttrName()))
    argConstraint = TensorKindInfo(TensorKind::Host);
  return argConstraint;
}

static TensorKindInfo getReturnedValueInfo(OpOperand &operand) {
  assert(operand.getOwner()->hasTrait<OpTrait::ReturnLike>() &&
         "expected ReturnLike owner");
  auto func = operand.getOwner()->getParentOfType<FunctionOpInterface>();
  assert(func && "expected parent FunctionOpInterface");
  if (func.getResultAttr(operand.getOperandNumber(),
                         getHostTensorArgAttrName()))
    return TensorKindInfo(TensorKind::Host);
  return TensorKindInfo(TensorKind::Device);
}

TensorKind TensorKindAnalysis::getStaticOperandTensorKind(OpOperand &operand) {
  Operation *op = operand.getOwner();
  if (auto iface = dyn_cast<TensorKindOpInterface>(op))
    return iface.getStaticOperandTensorKind(operand);

  if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(op))
    return operand.get() == reshapeOp.getShape() ? TensorKind::Host
                                                 : TensorKind::Unknown;
  if (auto extractOp = dyn_cast<tensor::ExtractOp>(op))
    return TensorKind::Host;

  if (op->hasTrait<OpTrait::ReturnLike>()) {
    auto parentFunc = dyn_cast<FunctionOpInterface>(op->getParentOp());
    if (!parentFunc)
      return TensorKind::Unknown;
    Region &bodyRegion = parentFunc.getFunctionBody();
    if (!bodyRegion.hasOneBlock() || op != bodyRegion.front().getTerminator())
      return TensorKind::Unknown;
    if (parentFunc.getResultAttr(operand.getOperandNumber(),
                                 getHostTensorArgAttrName()))
      return TensorKind::Host;
    return TensorKind::Device;
  }

  return TensorKind::Unknown;
}

LogicalResult TensorKindAnalysis::visitOperation(
    Operation *op, ArrayRef<TensorKindLattice *> operands,
    ArrayRef<const TensorKindLattice *> results) {

  auto setInferredType = [&](OpOperand &operand, TensorKindInfo kind) {
    assert(operand.getOwner() == op && "operand has the wrong owner");
    unsigned idx = operand.getOperandNumber();
    // If this is a function argument, first apply the required constraint info.
    // This is just required to ensure that function arguments get the type
    // "both" if a direct user requires a host tensor (e.g.
    // `tensor.extract`).
    if (isFunctionArgument(operand.get())) {
      TensorKindInfo argInfo = getFunctionArgInfo(operand.get());
      if (!argInfo.isUninitialized() && argInfo != kind)
        propagateIfChanged(operands[idx], operands[idx]->meet(argInfo));
    }
    return propagateIfChanged(operands[idx], operands[idx]->meet(kind));
  };

  if (auto opIface = dyn_cast<mlir::TensorKindOpInterface>(op)) {
    opIface.inferOperandKind(operands, results, setInferredType);
    return success();
  }

  if (auto tensorReshapeOp = dyn_cast<tensor::ReshapeOp>(op)) {
    // Propagate result kind to the input tensor kind.
    setInferredType(tensorReshapeOp.getSourceMutable(), results[0]->getValue());
    setInferredType(tensorReshapeOp.getShapeMutable(), TensorKind::Host);
    return success();
  }

  if (auto tensorExtractOp = dyn_cast<tensor::ExtractOp>(op)) {
    setInferredType(tensorExtractOp.getTensorMutable(), TensorKind::Host);
    return success();
  }

  if (auto tensorInsertOp = dyn_cast<tensor::InsertOp>(op)) {
    setInferredType(tensorInsertOp.getDestMutable(), TensorKind::Host);
    return success();
  }

  if (auto bufferizeOp = dyn_cast<bufferization::AllocTensorOp>(op)) {
    // It has no tensor operands, nothing to do.
    if (!bufferizeOp.getCopy() || !bufferizeOp.getMemorySpace()) {
      return success();
    }
    if (auto memSpace = dyn_cast_or_null<TensorKindAttrInterface>(
            bufferizeOp.getMemorySpaceAttr())) {
      if (memSpace.getTensorKind().isHostOnly()) {
        setInferredType(bufferizeOp.getCopyMutable()[0], TensorKind::Device);
        return success();
      }
      if (memSpace.getTensorKind().isDeviceOnly()) {
        setInferredType(bufferizeOp.getCopyMutable()[0], TensorKind::Host);
        return success();
      }
    }
  }

  // If an operation is not handled by the above special cases, then we use this
  // default implementation. The default implementation has two cases:
  //  - In the case where the operands are "small" (scalars or tensors of < 8
  //    elements): Each operand is the meet of every result. This propagates the
  //    result kinds up to the operands. So if the result kinds are all of a
  //    single kind (`host` or `device`), then the operands will have the same
  //    kind. If there is a mixture of kinds, then the operands will be `both`.
  if (llvm::all_of(op->getOperandTypes(), isScalarOrSmallTensor)) {
    for (const TensorKindLattice *result : results) {
      if (result->getValue().isUninitialized()) {
        // A result may be uninitialized if it is not used, so it can be safely
        // skipped.
        continue;
      }
      for (OpOperand &operand : op->getOpOperands()) {
        if (isa<RankedTensorType>(operand.get().getType()))
          setInferredType(operand, result->getValue());
      }
      addDependency(const_cast<TensorKindLattice *>(result),
                    getProgramPointAfter(op));
    }
    return success();
  }
  //  - Otherwise, at least one operand is not "small". We set operands to
  //    "device", regardless of result types. This has the effect of creating
  //    host/device boundaries where the result must be copied back to the host.
  //    Imagine, for example, that a loop condition (which always must be a host
  //    tensor), is the result of a very large reduction. In that case, we still
  //    want the reduction to run on the device, so we shouldn't propagate the
  //    'host' kind to the operands.
  for (OpOperand &operand : op->getOpOperands()) {
    // Conservatively handle non-tensor typed operands.
    if (!isa<ShapedType>(operand.get().getType()))
      continue;
    setInferredType(operand, TensorKind::Device);
  }
  return success();
}

void TensorKindAnalysis::setToExitState(TensorKindLattice *lattice) {
  Value v = lattice->getPoint();
  ChangeResult changed = lattice->meet(TensorKindInfo(TensorKind::Unknown));
  for (OpOperand &use : v.getUses()) {
    if (isReturnedValue(use)) {
      TensorKindInfo info = getReturnedValueInfo(use);
      changed |= lattice->meet(info);
    }
  }
  propagateIfChanged(lattice, changed);
}

void TensorKindAnalysis::visitBranchOperand(OpOperand &operand) {
  // Mark all non-forwarded branch operands as host tensors. This is in
  // line with the general strategy of requiring branching decisions to
  // be made on the host, but we could reevaluate this using device-side
  // graph/kernel launch.
  TensorKindLattice *lattice = getLatticeElement(operand.get());
  if (!isa<RankedTensorType>(lattice->getPoint().getType()))
    return;
  ChangeResult change = lattice->getValue().setKind(TensorKind::Host);
  propagateIfChanged(lattice, change);
}

void TensorKindAnalysis::visitCallOperand(OpOperand &operand) {
  // This function is called for non-forwarded call operands. An example of a
  // non-forwarded operand is something like a special call parameter. We always
  // just specify that these most be on the host.
#ifndef NDEBUG
  auto callOpInterface = dyn_cast<CallOpInterface>(operand.getOwner());
  assert(callOpInterface && "expected CallOpInterface");
  assert(
      !llvm::is_contained(callOpInterface.getArgOperandsMutable(), operand) &&
      "expected operand not to be a forwarded call argument");
#endif

  TensorKindLattice *lattice = getLatticeElement(operand.get());

  if (!isa<RankedTensorType>(operand.get().getType()))
    return;

  ChangeResult change = lattice->getValue().setKind(TensorKind::Host);
  propagateIfChanged(lattice, change);
}

void TensorKindAnalysis::visitExternalCall(
    CallOpInterface callOpInterface, ArrayRef<TensorKindLattice *> operands,
    ArrayRef<const TensorKindLattice *> results) {
  llvm::MutableArrayRef<OpOperand> forwardedOperands =
      callOpInterface.getArgOperandsMutable();

  llvm::SmallPtrSet<OpOperand *, 8> forwardedOperandsSet;
  for (OpOperand &operand : forwardedOperands)
    forwardedOperandsSet.insert(&operand);

  // This function is called for external calls. We always just specify that
  // these most be on the host.
  for (auto [lattice, operand] :
       llvm::zip_equal(operands, callOpInterface->getOpOperands())) {
    if (llvm::is_contained(forwardedOperandsSet, &operand)) {
      setToExitState(lattice);
      continue;
    }
    // This is a non-forwarded operand. We set it to host.
    visitCallOperand(operand);
  }
}
