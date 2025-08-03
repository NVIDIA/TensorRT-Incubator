//===- TensorRTRuntime.cpp - ----------------------------------------------===//
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
///
/// TensorRT runtime dialect implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::trtrt;

//===----------------------------------------------------------------------===//
// CompiledModuleOp
//===----------------------------------------------------------------------===//

LogicalResult CompiledFuncOp::verify() {
  auto dataType = dyn_cast<ShapedType>(getValue().getType());
  if (!dataType || !dataType.getElementType().isInteger(8) ||
      dataType.getRank() != 1)
    return emitOpError() << "expected data element type to be a 1D shaped type "
                            "with i8 element type";
  return success();
}

//===----------------------------------------------------------------------===//
// GetFunctionOp
//===----------------------------------------------------------------------===//

LogicalResult
GetFunctionOp::verifySymbolUses(SymbolTableCollection &collection) {
  if (!collection.lookupNearestSymbolFrom<trtrt::CompiledFuncOp>(
          getOperation(), getModuleAttr()))
    return emitError("failed to find TensorRT serialized engine symbol");
  return success();
}

//===----------------------------------------------------------------------===//
// EnqueueOp
//===----------------------------------------------------------------------===//

LogicalResult EnqueueOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  EnqueueOp::Adaptor adaptor(operands, attributes, properties, regions);

  // If the `outs` operands are tensor types, then we should return those as
  // results. Otherwise, for memref outs, we do not return results.
  for (Type t : TypeRange(adaptor.getOuts())) {
    auto tensorType = dyn_cast<TensorType>(t);
    if (!tensorType)
      continue;
    inferredReturnTypes.push_back(tensorType);
  }
  return success();
}

LogicalResult EnqueueOp::verify() {
  if (std::optional<ArrayRef<int64_t>> hostTensorIndices =
          getHostTensorArgs()) {
    // We don't count the context and stream argument here.
    const int64_t numInputArgs = getInputs().size();
    for (int64_t idx : *hostTensorIndices) {
      if (idx >= numInputArgs || idx < 0)
        return emitOpError("host_tensor_args value ")
               << idx << " is out of bounds";
      Value operand = getInputs()[idx];
      Type elType = mlir::getElementTypeOrSelf(operand.getType());
      if (!elType.isInteger(32))
        return emitOpError("host tensor arguments must have element type i32, "
                           "but input arg ")
               << idx << " has type " << operand.getType();
    }
  }
  return success();
}

void EnqueueOp::inferOperandKind(
    ArrayRef<TensorKindLattice *> operands,
    ArrayRef<const TensorKindLattice *> results,
    llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<ShapedType>(operand.get().getType()))
      continue;
    if (isOperandOnHost(&operand)) {
      setOperandKind(operand, TensorKind::Host);
      continue;
    }
    setOperandKind(operand, TensorKind::Device);
  }
}

TensorKind EnqueueOp::getStaticOperandTensorKind(OpOperand &operand) {
  return isOperandOnHost(&operand) ? TensorKind::Host : TensorKind::Device;
}

void EnqueueOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getInputsMutable()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : getOutsMutable()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// EnqueueAllocOp
//===----------------------------------------------------------------------===//

void EnqueueAllocOp::inferOperandKind(
    ArrayRef<TensorKindLattice *> operands,
    ArrayRef<const TensorKindLattice *> results,
    llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<ShapedType>(operand.get().getType()))
      continue;
    if (isOperandOnHost(&operand)) {
      setOperandKind(operand, TensorKind::Host);
      continue;
    }
    setOperandKind(operand, TensorKind::Device);
  }
}

TensorKind EnqueueAllocOp::getStaticOperandTensorKind(OpOperand &operand) {
  return isOperandOnHost(&operand) ? TensorKind::Host : TensorKind::Device;
}

LogicalResult EnqueueAllocOp::verify() {
  // Verify host tensor indices.
  if (std::optional<ArrayRef<int64_t>> hostTensorIndices =
          getHostTensorArgs()) {
    // We don't count the context and stream argument here.
    const int64_t numInputArgs = getInputs().size();
    for (int64_t idx : *hostTensorIndices) {
      if (idx >= numInputArgs || idx < 0)
        return emitOpError("host_tensor_args value ")
               << idx << " is out of bounds";
      Value operand = getInputs()[idx];
      Type elType = mlir::getElementTypeOrSelf(operand.getType());
      if (!elType.isInteger(32))
        return emitOpError("host tensor arguments must have element type i32, "
                           "but input arg ")
               << idx << " has type " << operand.getType();
    }
  }

  if (getNumResults() == 0)
    return emitOpError("at least one result is required.");

  bool isRankedTensor = isa<RankedTensorType>(getResult(0).getType());
  bool isMemRef = isa<MemRefType>(getResult(0).getType());

  // Verify that all results are either all tensors or all memrefs
  if (!isRankedTensor && !isMemRef)
    return emitOpError("result must be either RankedTensorType or MemRefType");

  for (auto resultType : getResultTypes()) {
    if (isRankedTensor && !isa<RankedTensorType>(resultType))
      return emitOpError("all results must be RankedTensorType");
    if (isMemRef && !isa<MemRefType>(resultType))
      return emitOpError("all results must be MemRefType");
    if (isMemRef && !dyn_cast<MemRefType>(resultType).getLayout().isIdentity())
      return emitOpError("result must have a canonical stride");
  }

  return success();
}

void EnqueueAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  for (OpOperand &operand : getInputsMutable()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
  }

  for (OpResult result : getResults()) {
    if (!isa<MemRefType>(result.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Write::get(), result);
    effects.emplace_back(MemoryEffects::Allocate::get(), result);
  }
}

//===----------------------------------------------------------------------===//
// TensorRTRuntimeDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with func operations.
struct TensorRTRuntimeInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// `tensorrt.enqueue` cannot be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return false;
  }

  /// All operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All functions can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Tablegen'd op definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOps.cpp.inc"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsTypes.cpp.inc"

void TensorRTRuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsTypes.cpp.inc"
      >();

  addInterfaces<TensorRTRuntimeInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface,
                           TensorRTRuntimeDialect>();
}
