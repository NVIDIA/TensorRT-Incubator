//===- PlanInterfaces.cpp -- ----------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Definitions for Plan op/attribute/type interfaces.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"

#define DEBUG_TYPE "plan-interfaces"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::plan;

#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.cpp.inc"

static int64_t getScalarBitWidth(Type type) {
  assert(type.isIntOrIndexOrFloat() && "expected scalar type");
  // Index type doesn't have a bitwidth, so we treat as 64 bit. This will allow
  // folding of conversions that are index casts to 32bit or 64bit.
  return isa<IndexType>(type) ? 64 : type.getIntOrFloatBitWidth();
}

static int64_t getElementTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type))
    return 2 * getScalarBitWidth(complexType.getElementType());
  if (auto vectorType = dyn_cast<VectorType>(type))
    return getScalarBitWidth(vectorType.getElementType()) *
           vectorType.getNumElements();
  return getScalarBitWidth(type);
}

/// Determines whether a cluster being outlined should clone a constant or
/// pass constant by value.
bool plan::detail::shouldCloneProducerDefault(Operation *producer,
                                              Region &targetRegion,
                                              bool allowTensorValuesOnly) {
  const bool isConstantLike = producer->hasTrait<OpTrait::ConstantLike>();
  if (!isConstantLike)
    return false;

  RankedTensorType type =
      dyn_cast<RankedTensorType>(producer->getResultTypes().front());
  if (!type)
    return !allowTensorValuesOnly &&
           isa<IndexType, IntegerType, FloatType, ComplexType>(
               producer->getResultTypes().front());

  // A value should be cloned if all of its uses are in the cluster.
  if (llvm::all_of(producer->getUsers(), [&](Operation *user) {
        return targetRegion.isAncestor(user->getParentRegion());
      }))
    return true;
  return type.getNumElements() *
             llvm::divideCeil(getElementTypeBitWidth(type.getElementType()),
                              8) <
         100 * 1024 * 1024;
}
