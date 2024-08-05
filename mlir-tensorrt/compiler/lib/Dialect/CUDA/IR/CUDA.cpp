//===- CUDA.cpp -----------------------------------------------------------===//
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

#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::cuda;

//===----------------------------------------------------------------------===//
// CompiledModuleOp
//===----------------------------------------------------------------------===//

LogicalResult CompiledModuleOp::verify() {
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
  if (!collection.lookupNearestSymbolFrom<cuda::CompiledModuleOp>(
          getOperation(), getModuleAttr()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  MemRefType type = getType();
  unsigned numDynamicDims = type.getNumDynamicDims();
  if (getDynamicSizes().size() != numDynamicDims)
    return emitOpError() << "number of dynamic size operands ("
                         << getDynamicSizes().size()
                         << ") should be equal to the number of dynamic "
                            "dimensions in the result type ("
                         << numDynamicDims << ")";

  if (std::optional<uint64_t> alignment = getAlignment()) {
    if (!llvm::isPowerOf2_64(*alignment)) {
      return emitOpError() << "alignment should be a power of 2 but got "
                           << *alignment;
    }
  }

  if (auto space =
          dyn_cast_or_null<executor::MemoryTypeAttr>(type.getMemorySpace())) {
    if (space.getValue() == executor::MemoryType::host_pinned) {
      if (getStream() || getDevice())
        return emitOpError()
               << "'stream' and 'device' arguments should not be specified "
                  "when the allocation type is host pinned memory";
    }
    if (space.getValue() == executor::MemoryType::host)
      return emitOpError()
             << "result memory space should be device or host_pinned";

    if (space.getValue() != executor::MemoryType::host_pinned &&
        (!getStream() || !getDevice()))
      return emitOpError() << "should have 'stream' and 'device' operands when "
                              "allocation is in device space";
  }

  if (auto space =
          dyn_cast_or_null<plan::MemorySpaceAttr>(type.getMemorySpace())) {
    if (space.getValue() == plan::MemorySpace::host_pinned) {
      if (getStream() || getDevice())
        return emitOpError()
               << "'stream' and 'device' arguments should not be specified "
                  "when the allocation type is host pinned memory";
    }
    if (space.getValue() == plan::MemorySpace::host)
      return emitOpError()
             << "result memory space should be device or host_pinned";

    if (space.getValue() != plan::MemorySpace::host_pinned &&
        (!getStream() || !getDevice()))
      return emitOpError() << "should have 'stream' and 'device' operands when "
                              "allocation is in device space";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOps
//===----------------------------------------------------------------------===//

static LogicalResult verifyCopyTypes(Operation *op, MemRefType sourceType,
                                     MemRefType targetType) {
  if (sourceType.hasStaticShape() && targetType.hasStaticShape() &&
      sourceType.getNumElements() != targetType.getNumElements())
    op->emitOpError() << llvm::formatv(
        "source and target memrefs must have the same number of elements"
        ", but the source type has {0} elements and the target type has {1} "
        "elements",
        sourceType.getNumElements(), targetType.getNumElements());

  return success();
}

LogicalResult CopyH2DOp::verify() {
  return verifyCopyTypes(getOperation(), getSource().getType(),
                         getTarget().getType());
}

LogicalResult CopyD2DOp::verify() {
  return verifyCopyTypes(getOperation(), getSource().getType(),
                         getTarget().getType());
}

LogicalResult CopyD2HOp::verify() {
  return verifyCopyTypes(getOperation(), getSource().getType(),
                         getTarget().getType());
}

//===----------------------------------------------------------------------===//
// MemSetOp
//===----------------------------------------------------------------------===//

LogicalResult MemSetOp::verify() {
  if (getFillValue().getType() != getMemref().getType().getElementType())
    return emitOpError()
           << "expected fill value type to match memref element type";
  int64_t bitwidth = getFillValue().getType().getIntOrFloatBitWidth();
  if (bitwidth != 32 && bitwidth != 8 && bitwidth != 16)
    return emitOpError()
           << "expected element type bit-width to be 8, 16, or 32";
  return success();
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : (*this)->getOpOperands()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// BlasHeuristicAlgoSelectionOp
//===----------------------------------------------------------------------===//

LogicalResult BlasHeuristicAlgoSelectionOp::verify() {
  if (ShapedType::isDynamicShape(getSizeA()) ||
      ShapedType::isDynamicShape(getStrideA()) ||
      ShapedType::isDynamicShape(getSizeB()) ||
      ShapedType::isDynamicShape(getStrideB()) ||
      ShapedType::isDynamicShape(getSizeC()) ||
      ShapedType::isDynamicShape(getStrideC()))
    return emitOpError("All attribute dimensions must be static.");
  return success();
}

//===----------------------------------------------------------------------===//
// BlasRunGemmOp
//===----------------------------------------------------------------------===//

static LogicalResult isRankValidForMatmul(ShapedType t) {
  if (t.getRank() != 2 && t.getRank() != 3)
    return failure();
  return success();
}

static LogicalResult isLastDimContiguous(MemRefType t) {
  if (mlir::getStridesAndOffset(t).first.back() != 1)
    return failure();
  return success();
}

LogicalResult BlasRunGemmOp::verify() {
  if (!getAlpha() && getBeta())
    return emitOpError("For GEMM, both `alpha` and `beta` should be provided.");

  // If `mat_a` is a tensor type, make sure all other inputs are tensors as well
  if (auto tensorType = dyn_cast<TensorType>(getMatA().getType())) {
    if (!dyn_cast<TensorType>(getMatB().getType()) ||
        !dyn_cast<TensorType>(getMatC().getType()))
      return emitOpError(
          "If one input is `TensorType`, all must be `TensorType`.");
  }

  // If `mat_a` is a memref type, make sure all other inputs are memref as well
  if (auto tensorType = dyn_cast<MemRefType>(getMatA().getType())) {
    if (!dyn_cast<MemRefType>(getMatB().getType()) ||
        !dyn_cast<MemRefType>(getMatC().getType()))
      return emitOpError(
          "If first input is `MemRefType`, all must be `MemRefType`.");

    if (!getAlgo())
      return emitOpError("If input is `MemRefType`, `algo` must be provided.");

    // cuBLAS needs last dim of every input to be contiguous
    if (failed(
            isLastDimContiguous(dyn_cast<MemRefType>(getMatA().getType()))) ||
        failed(
            isLastDimContiguous(dyn_cast<MemRefType>(getMatB().getType()))) ||
        failed(isLastDimContiguous(dyn_cast<MemRefType>(getMatC().getType()))))
      return emitOpError(
          "cuBLAS needs last dimension to be contiguous i.e. stride 1");
  }

  // Check if input is valid for matmul
  if (failed(isRankValidForMatmul(cast<ShapedType>(getMatA().getType()))) ||
      failed(isRankValidForMatmul(cast<ShapedType>(getMatB().getType()))) ||
      failed(isRankValidForMatmul(cast<ShapedType>(getMatC().getType()))))
    return emitOpError("All inputs must be 2D or 3D (first batch dim) for "
                       "cuBLAS GEMM/MatMul.");

  // Check for data type support
  // Only FP16, FP32, F64, and I32 types are supported
  for (auto t : getOperandTypes()) {
    auto maybeShapedType = dyn_cast<ShapedType>(t);
    if (maybeShapedType && !maybeShapedType.getElementType().isF16() &&
        !maybeShapedType.getElementType().isF32() &&
        !maybeShapedType.getElementType().isF64() &&
        !maybeShapedType.getElementType().isInteger(32))
      return emitOpError(
          "Currently, only FP16, FP32, F64, and I32 types are supported.");
  }
  return success();
}

LogicalResult BlasRunGemmOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  BlasRunGemmOp::Adaptor adaptor(operands, attributes, properties, regions);

  // If the `out` operand is a tensor type, then we should return that as
  // results. Otherwise, for memref types, we do not return results.
  if (auto tensorType = dyn_cast<TensorType>(adaptor.getMatC().getType()))
    inferredReturnTypes.push_back(tensorType);
  return success();
}

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ArrayRef<OpOperand *> inputOperands,
    MutableOperandRange outputOperands) {
  for (OpOperand *operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand->get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

void BlasRunGemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(effects, getOperation()->getResults(), getDpsInputOperands(),
                 getDpsInitsMutable());
}
//===----------------------------------------------------------------------===//
// CUDADialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining.
struct CUDAInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return false;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void CUDADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsTypes.cpp.inc"
      >();

  addInterfaces<CUDAInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOps.cpp.inc"
