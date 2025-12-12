//===- GenerateSortValueWrapper.h -----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Wrapper classes for fluent MLIR operation building in merge sort kernels.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_KERNEL_TRANSFORMS_GENERATESORTVALUEWRAPPER_H
#define MLIR_KERNEL_TRANSFORMS_GENERATESORTVALUEWRAPPER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include <limits>

namespace mlir {
namespace kernel {

// Forward declarations
class ValueWrapper;
class ConditionWrapper;
class MemRefWrapper;
class Context;

/// ValueWrapper provides a fluent interface for building MLIR arithmetic
/// operations that mirrors the original C++ algorithm syntax
class ValueWrapper {
  OpBuilder *builder;
  Location loc;
  Value value;

public:
  ValueWrapper(OpBuilder &b, Location l, Value v)
      : builder(&b), loc(l), value(v) {}

  // Arithmetic operators (assume signed semantics)
  ValueWrapper operator+(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::AddIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator-(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::SubIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator*(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::MulIOp>(loc, value, rhs.value));
  }

  /// Division operator - uses signed division semantics
  ValueWrapper operator/(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::DivSIOp>(loc, value, rhs.value));
  }

  /// Explicit unsigned division for cases where unsigned semantics are needed
  ValueWrapper divU(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::DivUIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator&(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::AndIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator|(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::OrIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator^(const ValueWrapper &rhs) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::XOrIOp>(loc, value, rhs.value));
  }

  ValueWrapper operator~() const {
    auto minusOne = builder->create<arith::ConstantOp>(
        loc, value.getType(), builder->getIntegerAttr(value.getType(), -1));
    return ValueWrapper(*builder, loc,
                        builder->create<arith::XOrIOp>(loc, value, minusOne));
  }

  // Comparison operators
  ConditionWrapper operator<(const ValueWrapper &rhs) const;
  ConditionWrapper operator<=(const ValueWrapper &rhs) const;
  ConditionWrapper operator>(const ValueWrapper &rhs) const;
  ConditionWrapper operator>=(const ValueWrapper &rhs) const;
  ConditionWrapper operator==(const ValueWrapper &rhs) const;
  ConditionWrapper operator!=(const ValueWrapper &rhs) const;

  // Type conversions
  ValueWrapper toIndex() const {
    if (mlir::isa<IndexType>(value.getType()))
      return *this;
    return ValueWrapper(*builder, loc,
                        builder->create<arith::IndexCastOp>(
                            loc, builder->getIndexType(), value));
  }

  ValueWrapper toType(Type targetType) const {
    if (value.getType() == targetType)
      return *this;
    if (value.getType().isIndex() || targetType.isIndex())
      return ValueWrapper(
          *builder, loc,
          builder->create<arith::IndexCastOp>(loc, targetType, value));
    llvm_unreachable("Unsupported cast type");
  }

  // Convert back to mlir::Value
  operator Value() const { return value; }
  Value getValue() const { return value; }
  Location getLoc() const { return loc; }
  OpBuilder &getBuilder() const { return *builder; }
};

/// ConditionWrapper represents boolean conditions and provides select
/// operations
class ConditionWrapper {
  OpBuilder *builder;
  Location loc;
  Value condition;

public:
  ConditionWrapper(OpBuilder &b, Location l, Value c)
      : builder(&b), loc(l), condition(c) {}

  ValueWrapper select(const ValueWrapper &trueVal,
                      const ValueWrapper &falseVal) const {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::SelectOp>(
                            loc, condition, Value(trueVal), Value(falseVal)));
  }

  ConditionWrapper operator&(const ConditionWrapper &rhs) const {
    return ConditionWrapper(
        *builder, loc,
        builder->create<arith::AndIOp>(loc, condition, rhs.condition));
  }

  ConditionWrapper operator|(const ConditionWrapper &rhs) const {
    return ConditionWrapper(
        *builder, loc,
        builder->create<arith::OrIOp>(loc, condition, rhs.condition));
  }

  ConditionWrapper operator!() const {
    auto trueVal = builder->create<arith::ConstantOp>(
        loc, builder->getI1Type(), builder->getBoolAttr(true));
    return ConditionWrapper(
        *builder, loc, builder->create<arith::XOrIOp>(loc, condition, trueVal));
  }

  operator Value() const { return condition; }
  Value getValue() const { return condition; }
};

// Implement comparison operators that return ConditionWrapper
inline ConditionWrapper ValueWrapper::operator<(const ValueWrapper &rhs) const {
  auto predicate = value.getType().isUnsignedInteger()
                       ? arith::CmpIPredicate::ult
                       : arith::CmpIPredicate::slt;
  return ConditionWrapper(
      *builder, loc,
      builder->create<arith::CmpIOp>(loc, predicate, value, rhs.value));
}

inline ConditionWrapper
ValueWrapper::operator<=(const ValueWrapper &rhs) const {
  auto predicate = value.getType().isUnsignedInteger()
                       ? arith::CmpIPredicate::ule
                       : arith::CmpIPredicate::sle;
  return ConditionWrapper(
      *builder, loc,
      builder->create<arith::CmpIOp>(loc, predicate, value, rhs.value));
}

inline ConditionWrapper ValueWrapper::operator>(const ValueWrapper &rhs) const {
  auto predicate = value.getType().isUnsignedInteger()
                       ? arith::CmpIPredicate::ugt
                       : arith::CmpIPredicate::sgt;
  return ConditionWrapper(
      *builder, loc,
      builder->create<arith::CmpIOp>(loc, predicate, value, rhs.value));
}

inline ConditionWrapper
ValueWrapper::operator>=(const ValueWrapper &rhs) const {
  auto predicate = value.getType().isUnsignedInteger()
                       ? arith::CmpIPredicate::uge
                       : arith::CmpIPredicate::sge;
  return ConditionWrapper(
      *builder, loc,
      builder->create<arith::CmpIOp>(loc, predicate, value, rhs.value));
}

inline ConditionWrapper
ValueWrapper::operator==(const ValueWrapper &rhs) const {
  return ConditionWrapper(*builder, loc,
                          builder->create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::eq, value, rhs.value));
}

inline ConditionWrapper
ValueWrapper::operator!=(const ValueWrapper &rhs) const {
  return ConditionWrapper(*builder, loc,
                          builder->create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::ne, value, rhs.value));
}

/// MemRefWrapper provides easy memref operations
class MemRefWrapper {
  OpBuilder *builder;
  Location loc;
  Value memref;

public:
  MemRefWrapper(OpBuilder &b, Location l, Value m)
      : builder(&b), loc(l), memref(m) {}

  ValueWrapper load(const ValueWrapper &index) const {
    Value idx = index.toIndex();
    Value loaded = builder->create<memref::LoadOp>(loc, memref, idx);
    return ValueWrapper(*builder, loc, loaded);
  }

  ValueWrapper load(std::initializer_list<ValueWrapper> indices) const {
    SmallVector<Value> idxValues;
    for (const auto &idx : indices) {
      idxValues.push_back(idx.toIndex());
    }
    Value loaded = builder->create<memref::LoadOp>(loc, memref, idxValues);
    return ValueWrapper(*builder, loc, loaded);
  }

  void store(const ValueWrapper &value, const ValueWrapper &index) const {
    Value idx = index.toIndex();
    builder->create<memref::StoreOp>(loc, value.getValue(), memref, idx);
  }

  void store(const ValueWrapper &value,
             std::initializer_list<ValueWrapper> indices) const {
    SmallVector<Value> idxValues;
    for (const auto &idx : indices) {
      idxValues.push_back(idx.toIndex());
    }
    builder->create<memref::StoreOp>(loc, value.getValue(), memref, idxValues);
  }

  // Create a subview with offset
  MemRefWrapper subview(const ValueWrapper &offset, const ValueWrapper &size,
                        const ValueWrapper &stride) const {
    auto subviewOp = builder->create<memref::SubViewOp>(
        loc, memref, ArrayRef<OpFoldResult>{offset.toIndex().getValue()},
        ArrayRef<OpFoldResult>{size.toIndex().getValue()},
        ArrayRef<OpFoldResult>{stride.toIndex().getValue()});
    return MemRefWrapper(*builder, loc, subviewOp);
  }

  operator Value() const { return memref; }
  Value getValue() const { return memref; }
};

/// Context provides helper functions and manages the builder state
class Context {
  OpBuilder *builder;
  Location loc;

public:
  Context(OpBuilder &b, Location l) : builder(&b), loc(l) {}

  // Constant creation
  ValueWrapper constant(int64_t val) {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::ConstantIndexOp>(loc, val));
  }

  ValueWrapper constantInt(int64_t val, Type type) {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::ConstantOp>(
                            loc, type, builder->getIntegerAttr(type, val)));
  }

  ValueWrapper constantI32(int32_t val) {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::ConstantOp>(
                            loc, builder->getI32IntegerAttr(val)));
  }

  ValueWrapper constantBool(bool val) {
    return ValueWrapper(
        *builder, loc,
        builder->create<arith::ConstantOp>(loc, builder->getI1Type(),
                                           builder->getBoolAttr(val)));
  }

  ValueWrapper constantFloat(double val, Type type) {
    return ValueWrapper(*builder, loc,
                        builder->create<arith::ConstantOp>(
                            loc, type, builder->getFloatAttr(type, val)));
  }

  /// Create a sentinel value (maximum value) for the given type.
  /// Used in merge sort to represent exhausted sequences.
  /// For integers, returns the maximum signed value for the bit width.
  /// For floats, returns positive infinity.
  ValueWrapper sentinelValue(Type type) {
    if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
      // Compute maximum signed value for this bit width
      unsigned width = intType.getWidth();
      // Maximum signed value is 2^(width-1) - 1
      // For safety, cap at 63 bits to avoid int64_t overflow
      assert(width > 0 && width <= 64 && "integer width must be 1-64 bits");
      int64_t maxVal;
      if (width >= 64)
        maxVal = std::numeric_limits<int64_t>::max();
      else
        maxVal = (1LL << (width - 1)) - 1;
      return constantInt(maxVal, type);
    } else if (mlir::isa<FloatType>(type)) {
      // For float types, use positive infinity
      return constantFloat(std::numeric_limits<double>::infinity(), type);
    } else {
      llvm_unreachable("Unsupported type for sentinel value");
    }
  }

  // Min/max operations
  ValueWrapper min(const ValueWrapper &a, const ValueWrapper &b) {
    auto type = a.getValue().getType();
    Value result;
    if (type.isUnsignedInteger()) {
      result = builder->create<arith::MinUIOp>(loc, Value(a), Value(b));
    } else {
      result = builder->create<arith::MinSIOp>(loc, Value(a), Value(b));
    }
    return ValueWrapper(*builder, loc, result);
  }

  ValueWrapper max(const ValueWrapper &a, const ValueWrapper &b) {
    auto type = a.getValue().getType();
    Value result;
    if (type.isUnsignedInteger()) {
      result = builder->create<arith::MaxUIOp>(loc, Value(a), Value(b));
    } else {
      result = builder->create<arith::MaxSIOp>(loc, Value(a), Value(b));
    }
    return ValueWrapper(*builder, loc, result);
  }

  // Memory allocation
  MemRefWrapper allocaDynamic(Type elementType, const ValueWrapper &size) {
    SmallVector<int64_t, 1> shape;
    shape.push_back(ShapedType::kDynamic);
    auto memrefType = MemRefType::get(shape, elementType);
    SmallVector<Value, 1> dynamicSizes;
    dynamicSizes.push_back(size.getValue());
    Value alloc =
        builder->create<memref::AllocaOp>(loc, memrefType, dynamicSizes);
    return MemRefWrapper(*builder, loc, alloc);
  }

  MemRefWrapper allocaStatic(Type elementType, int64_t size) {
    SmallVector<int64_t, 1> shape;
    shape.push_back(size);
    auto memrefType = MemRefType::get(shape, elementType);
    Value alloc = builder->create<memref::AllocaOp>(loc, memrefType);
    return MemRefWrapper(*builder, loc, alloc);
  }

  MemRefWrapper allocShared(Type elementType, int64_t size) {
    Attribute smemSpace = gpu::AddressSpaceAttr::get(
        builder->getContext(), gpu::AddressSpace::Workgroup);
    SmallVector<int64_t, 1> shape;
    shape.push_back(size);
    auto memrefType = MemRefType::get(shape, elementType,
                                      MemRefLayoutAttrInterface{}, smemSpace);
    Value alloc = builder->create<memref::AllocOp>(loc, memrefType);
    return MemRefWrapper(*builder, loc, alloc);
  }

  // Comparison helpers for different types
  ConditionWrapper compareKeys(const ValueWrapper &lhs, const ValueWrapper &rhs,
                               Type keyType) {
    if (mlir::isa<IntegerType>(keyType)) {
      return lhs < rhs;
    } else if (mlir::isa<FloatType>(keyType)) {
      return ConditionWrapper(
          *builder, loc,
          builder->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
                                         lhs.getValue(), rhs.getValue()));
    }
    llvm_unreachable("Unsupported key type for comparison");
  }

  // Control flow builders
  template <typename CondFn, typename BodyFn>
  ValueWrapper buildWhile(const ValueWrapper &init, CondFn condFn,
                          BodyFn bodyFn) {
    auto whileOp = builder->create<scf::WhileOp>(
        loc, TypeRange{init.getValue().getType()}, ValueRange{init.getValue()});

    // Before region
    {
      OpBuilder::InsertionGuard guard(*builder);
      Block *before = &whileOp.getBefore().emplaceBlock();
      before->addArgument(init.getValue().getType(), loc);
      builder->setInsertionPointToStart(before);

      ValueWrapper arg(*builder, loc, before->getArgument(0));
      auto condition = condFn(arg);
      builder->create<scf::ConditionOp>(loc, condition.getValue(),
                                        ValueRange{arg.getValue()});
    }

    // After region
    {
      OpBuilder::InsertionGuard guard(*builder);
      Block *after = &whileOp.getAfter().emplaceBlock();
      after->addArgument(init.getValue().getType(), loc);
      builder->setInsertionPointToStart(after);

      ValueWrapper arg(*builder, loc, after->getArgument(0));
      auto result = bodyFn(arg);
      builder->create<scf::YieldOp>(loc, ValueRange{result.getValue()});
    }

    return ValueWrapper(*builder, loc, whileOp.getResult(0));
  }

  template <typename BodyFn>
  void buildFor(const ValueWrapper &start, const ValueWrapper &end,
                const ValueWrapper &step, BodyFn bodyFn) {
    auto forOp = builder->create<scf::ForOp>(loc, start.getValue(),
                                             end.getValue(), step.getValue());
    {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(forOp.getBody());
      ValueWrapper inductionVar(*builder, loc, forOp.getInductionVar());
      bodyFn(inductionVar);
    }
  }

  template <typename BodyFn>
  auto buildForWithState(const ValueWrapper &start, const ValueWrapper &end,
                         const ValueWrapper &step, ValueRange initVals,
                         BodyFn bodyFn) {
    auto forOp = builder->create<scf::ForOp>(
        loc, start.getValue(), end.getValue(), step.getValue(), initVals);
    {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(forOp.getBody());
      ValueWrapper inductionVar(*builder, loc, forOp.getInductionVar());

      SmallVector<ValueWrapper> args;
      for (auto arg : forOp.getRegionIterArgs()) {
        args.emplace_back(*builder, loc, arg);
      }

      auto results = bodyFn(inductionVar, args);
      SmallVector<Value> yieldValues;
      for (const auto &res : results) {
        yieldValues.push_back(res.getValue());
      }
      builder->create<scf::YieldOp>(loc, yieldValues);
    }

    SmallVector<ValueWrapper> results;
    for (auto res : forOp.getResults()) {
      results.emplace_back(*builder, loc, res);
    }
    return results;
  }

  template <typename ThenFn>
  void buildIf(const ConditionWrapper &condition, ThenFn thenFn) {
    auto ifOp = builder->create<scf::IfOp>(loc, condition.getValue(), false);
    {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
      thenFn();
    }
  }

  template <typename ThenFn, typename ElseFn>
  void buildIfElse(const ConditionWrapper &condition, ThenFn thenFn,
                   ElseFn elseFn) {
    auto ifOp = builder->create<scf::IfOp>(loc, condition.getValue(), true);
    {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
      thenFn();
    }
    {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(&ifOp.getElseRegion().front());
      elseFn();
    }
  }

  // GPU specific helpers
  ValueWrapper threadId(gpu::Dimension dim = gpu::Dimension::x) {
    return ValueWrapper(*builder, loc,
                        builder->create<gpu::ThreadIdOp>(
                            loc, builder->getIndexType(), dim))
        .toType(builder->getI32Type());
  }

  ValueWrapper blockId(gpu::Dimension dim = gpu::Dimension::x) {
    return ValueWrapper(*builder, loc,
                        builder->create<gpu::BlockIdOp>(
                            loc, builder->getIndexType(), dim))
        .toType(builder->getI32Type());
  }

  ValueWrapper blockDim(gpu::Dimension dim = gpu::Dimension::x) {
    return ValueWrapper(*builder, loc,
                        builder->create<gpu::BlockDimOp>(
                            loc, builder->getIndexType(), dim))
        .toType(builder->getI32Type());
  }

  void syncThreads() { builder->create<gpu::BarrierOp>(loc); }
};

} // namespace kernel
} // namespace mlir

#endif // MLIR_KERNEL_TRANSFORMS_GENERATESORTVALUEWRAPPER_H
