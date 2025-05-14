//===- MemRefDescriptorBuilder.h --------------------------------*- C++ -*-===//
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
/// Defines an interface for lowering memref to an aggregate-type object.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_UTILS_MEMREFDESCRIPTORADAPTOR_H
#define MLIR_EXECUTOR_UTILS_MEMREFDESCRIPTORADAPTOR_H

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include <functional>

namespace mlir {

/// Represents a class for constructing struct-like aggregates.
class AggregateBuilder {
public:
  /// Builds IR to extract the `pos`-th item from `aggregate`.
  using ExtractBuilderType = llvm::function_ref<Value(
      OpBuilder &b, Location loc, Value aggregate, unsigned pos)>;

  /// Builds IR to insert `itemValue` into the `pos`-th item of `aggregate`,
  /// returning the value for the new or updated aggregate.
  using InsertBuilderType =
      llvm::function_ref<Value(OpBuilder &b, Location loc, Value aggregate,
                               Value itemValue, unsigned pos)>;

  /// Construct from an exisiting value representing the aggregate.
  AggregateBuilder(Value value, ExtractBuilderType extractBuilder,
                   InsertBuilderType insertBuilder)
      : value(value), aggregateType(value.getType()),
        extractValueBuilderFn(extractBuilder),
        insertValueBuilderFn(insertBuilder) {}

  virtual ~AggregateBuilder() {}

  /// Allow implicit conversion to Value.
  operator Value() { return value; }

  /// Return the type of the aggregate.
  Type getType() { return aggregateType; }

  /// Inserts `itemValue` into the `pos`-th position of the internal value and
  /// updates the internal value.
  virtual void insertValue(ImplicitLocOpBuilder &b, unsigned pos,
                           Value itemValue) {
    value = insertValueBuilderFn(b, b.getLoc(), value, itemValue, pos);
  }
  virtual void insertValue(OpBuilder &b, Location loc, unsigned pos,
                           Value itemValue) {
    value = insertValueBuilderFn(b, loc, value, itemValue, pos);
  }

  /// Builds IR to extract the `pos`-th item from `value`.
  virtual Value extractValue(ImplicitLocOpBuilder &b, unsigned pos) const {
    return extractValueBuilderFn(b, b.getLoc(), value, pos);
  }
  virtual Value extractValue(OpBuilder &b, Location loc, unsigned pos) const {
    return extractValueBuilderFn(b, loc, value, pos);
  }

protected:
  Value value;
  Type aggregateType;
  ExtractBuilderType extractValueBuilderFn;
  InsertBuilderType insertValueBuilderFn;
};

/// An interface used to construct and manipulate aggregate objects which
/// represent mermef descriptors. This class is adapted from the
/// MemrefDescriptor class in upstream LLVMCommon conversion utilities. We make
/// some specific adjustments: we don't carry 'alllocated' ptr separately. We
/// always assume that the descriptor is a flat table (i.e. shape and strides
/// are not maintained as separate arrays).
class MemRefDescriptorAdaptor : public AggregateBuilder {
public:
  static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 0;
  static constexpr unsigned kAlignedPtrPosInMemRefDescriptor = 1;
  static constexpr unsigned kOffsetPosInMemRefDescriptor = 2;
  static constexpr unsigned kSizePosInMemRefDescriptor = 3;

  using ConstantBuilderType =
      std::function<Value(OpBuilder &, Location, Type, int64_t)>;

  MemRefDescriptorAdaptor(Value descriptor, MemRefType memrefType,
                          ExtractBuilderType extractBuilder,
                          InsertBuilderType insertBuilder,
                          ConstantBuilderType constantBuilder, Type indexType)
      : AggregateBuilder(descriptor, extractBuilder, insertBuilder),
        constantBuilder(constantBuilder), indexType(indexType),
        memrefType(memrefType) {}

  // Analagous to 'pack'
  MemRefDescriptorAdaptor(ImplicitLocOpBuilder &b, Value undef,
                          MemRefType memrefType, ValueRange values,
                          ExtractBuilderType extractBuilder,
                          InsertBuilderType insertBuilder,
                          ConstantBuilderType constantBuilder, Type indexType)
      : AggregateBuilder(undef, extractBuilder, insertBuilder),
        constantBuilder(constantBuilder), indexType(indexType),
        memrefType(memrefType) {
    setAllocatedPtr(b, values[kAllocatedPtrPosInMemRefDescriptor]);
    setAlignedPtr(b, values[kAlignedPtrPosInMemRefDescriptor]);
    setOffset(b, values[kOffsetPosInMemRefDescriptor]);
    int64_t rank = memrefType.getRank();
    for (unsigned i = 0; i < rank; ++i) {
      setSize(b, i, values[kSizePosInMemRefDescriptor + i]);
      setStride(b, i, values[kSizePosInMemRefDescriptor + rank + i]);
    }
  }

  /// Builds IR extracting the allocated pointer from the descriptor.
  Value allocatedPtr(OpBuilder &b, Location loc) const {
    return extractValue(b, loc, kAllocatedPtrPosInMemRefDescriptor);
  }
  Value allocatedPtr(ImplicitLocOpBuilder &b) const {
    return allocatedPtr(b, b.getLoc());
  }

  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &b, Location loc, Value ptr) {
    insertValue(b, loc, kAllocatedPtrPosInMemRefDescriptor, ptr);
  }
  void setAllocatedPtr(ImplicitLocOpBuilder &b, Value ptr) {
    setAllocatedPtr(b, b.getLoc(), ptr);
  }

  /// Builds IR extracting the aligned pointer from the descriptor.
  Value alignedPtr(OpBuilder &b, Location loc) const {
    return extractValue(b, loc, kAlignedPtrPosInMemRefDescriptor);
  }
  Value alignedPtr(ImplicitLocOpBuilder &b) const {
    return alignedPtr(b, b.getLoc());
  }

  /// Builds IR inserting the aligned pointer into the descriptor.
  void setAlignedPtr(ImplicitLocOpBuilder &b, Value ptr) {
    insertValue(b, kAlignedPtrPosInMemRefDescriptor, ptr);
  }

  /// Builds IR extracting the offset from the descriptor.
  Value offset(OpBuilder &b, Location loc) const {
    auto [strides, offset] =
        const_cast<MemRefType &>(memrefType).getStridesAndOffset();
    if (!ShapedType::isDynamic(offset))
      return constantBuilder(b, loc, indexType, offset);
    return extractValue(b, loc, kOffsetPosInMemRefDescriptor);
  }
  Value offset(ImplicitLocOpBuilder &b) const { return offset(b, b.getLoc()); }

  /// Builds IR inserting the offset into the descriptor.
  void setOffset(ImplicitLocOpBuilder &b, Value offset) {
    insertValue(b, kOffsetPosInMemRefDescriptor, offset);
  }

  /// Builds IR extracting the pos-th size from the descriptor.
  virtual Value size(OpBuilder &b, Location loc, unsigned pos) const {
    if (memrefType.getDimSize(pos) != ShapedType::kDynamic)
      return constantBuilder(b, loc, indexType, memrefType.getDimSize(pos));
    return extractValue(b, loc, kSizePosInMemRefDescriptor + pos);
  }
  Value size(ImplicitLocOpBuilder &b, unsigned pos) const {
    return size(b, b.getLoc(), pos);
  }

  /// Builds IR inserting the pos-th size into the descriptor
  virtual void setSize(ImplicitLocOpBuilder &b, unsigned pos, Value size) {
    return insertValue(b, kSizePosInMemRefDescriptor + pos, size);
  }

  /// Builds IR extracting the pos-th size from the descriptor.
  virtual Value stride(OpBuilder &b, Location loc, unsigned pos) const {
    return extractValue(
        b, loc, kSizePosInMemRefDescriptor + memrefType.getRank() + pos);
  }
  Value stride(ImplicitLocOpBuilder &b, unsigned pos) const {
    return stride(b, b.getLoc(), pos);
  }

  /// Builds IR inserting the pos-th stride into the descriptor
  virtual void setStride(ImplicitLocOpBuilder &b, unsigned pos, Value stride) {
    return insertValue(
        b, kSizePosInMemRefDescriptor + memrefType.getRank() + pos, stride);
  }

  static unsigned getDescriptorByteSize(MemRefType memrefType,
                                        unsigned pointerSizeBytes,
                                        unsigned indexTypeByteSize) {
    return 2 * pointerSizeBytes +
           (1 + 2 * memrefType.getRank()) * indexTypeByteSize;
  }

  static unsigned getNumDescriptorFields(MemRefType memrefType) {
    return 3 + 2 * memrefType.getRank();
  }

  void setSizes(ImplicitLocOpBuilder &b, ValueRange sizes) {
    assert(static_cast<int64_t>(sizes.size()) == memrefType.getRank());
    for (auto [it, val] : llvm::enumerate(sizes))
      setSize(b, it, val);
  }

  void setStrides(ImplicitLocOpBuilder &b, ValueRange strides) {
    assert(static_cast<int64_t>(strides.size()) == memrefType.getRank());
    for (auto [it, val] : llvm::enumerate(strides))
      setStride(b, it, val);
  }

  MemRefType getMemRefType() const {
    assert(memrefType && "expected valid type");
    return memrefType;
  }

protected:
  ConstantBuilderType constantBuilder;
  Type indexType;
  MemRefType memrefType{nullptr};
};

} // namespace mlir

#endif // MLIR_EXECUTOR_UTILS_MEMREFDESCRIPTORADAPTOR_H
