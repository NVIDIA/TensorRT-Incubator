//===- ExpandMemRefArgs.cpp -----------------------------------------------===//
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
///===- ExpandMemRefArgs.cpp ----------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This file implements the KernelExpandMemRefArgsPass pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_KERNELEXPANDMEMREFARGSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

namespace {

/// Encapsulates the individual components of a MemRef object.
struct ResolvedMetadata {
  Value base;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  SmallVector<Value> getDynamicValues() const {
    SmallVector<Value> result = {base};
    if (isa<Value>(offset))
      result.push_back(cast<Value>(offset));
    for (auto size : sizes)
      if (isa<Value>(size))
        result.push_back(cast<Value>(size));
    for (auto stride : strides)
      if (isa<Value>(stride))
        result.push_back(cast<Value>(stride));
    return result;
  }
};

/// Type converter that converts MemRef types to their expanded form.
/// For Kernel memref arguments, we have a particular scheme for converting
/// memref arguments. We convert to:
/// - aligned pointer (in the form of a 0-rank memref)
/// - offset (if not statically known)
/// - shape (only non-statically known dimensions)
/// - strides (only non-statically known dimensions)
///
/// We then use the "bare-pointer" calling convention to lower the function
/// during the lowering to the lower-level target. For 0-rank memrefs, this
/// always expands to the single aligned pointer.
struct MemRefTypeConverter : public TypeConverter {
  MemRefTypeConverter(MLIRContext *context) : context(context) {
    addConversion([&](Type type) -> Type { return type; });
    addConversion([&](MemRefType type, SmallVectorImpl<Type> &types) {
      return convertMemRefType(type, types);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
        FailureOr<Value> result =
            this->reconstructMemRef(builder, loc, inputs, memrefType);
        if (failed(result))
          return {};
        return *result;
      }
      return {};
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
      if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
        FailureOr<Value> result =
            this->reconstructMemRef(builder, loc, inputs, memrefType);
        if (failed(result))
          return {};
        return *result;
      }
      return {};
    });

    addTargetMaterialization([&](OpBuilder &builder, TypeRange resultTypes,
                                 ValueRange inputs, Location loc,
                                 Type originalType) -> SmallVector<Value> {
      if (auto memrefType = dyn_cast<MemRefType>(originalType)) {
        if (inputs.size() != 1)
          return {};
        FailureOr<ResolvedMetadata> metadata = this->decomposeMemRef(
            builder, loc, cast<TypedValue<MemRefType>>(inputs[0]));
        if (failed(metadata))
          return {};
        return metadata->getDynamicValues();
      }
      return {};
    });
  }

  /// Converts a MemRef type to list of types that represent the expanded form.
  /// Adds the 0-rank memref representing the aligned pointer, and appends one
  /// index type for each dynamic value.
  LogicalResult convertMemRefType(MemRefType type,
                                  SmallVectorImpl<Type> &result) const {
    /// Add the 0-rank memref representing the aligned pointer.
    result.push_back(MemRefType::get({}, type.getElementType(),
                                     MemRefLayoutAttrInterface{},
                                     type.getMemorySpace()));

    // Extract aligned metadata. We append one index type for each dynamic
    // value.
    int64_t offset = 0;
    SmallVector<int64_t, 4> strides;
    if (failed(type.getStridesAndOffset(strides, offset)))
      return failure();

    if (ShapedType::isDynamic(offset))
      result.push_back(indexType);
    for (int64_t d : type.getShape())
      if (ShapedType::isDynamic(d))
        result.push_back(indexType);
    for (int64_t d : strides)
      if (ShapedType::isDynamic(d))
        result.push_back(indexType);

    return success();
  }

  /// Resolves the metadata from the expanded form.
  ResolvedMetadata resolveMetadata(ValueRange values, int64_t offset,
                                   ArrayRef<int64_t> shape,
                                   ArrayRef<int64_t> strides) {
    ResolvedMetadata result;
    result.base = values[0];
    values = values.drop_front();

    if (ShapedType::isDynamic(offset)) {
      result.offset = values.front();
      values = values.drop_front();
    } else {
      result.offset = IntegerAttr::get(indexType, offset);
    }

    auto handleElement = [&](int64_t d) -> OpFoldResult {
      if (ShapedType::isDynamic(d)) {
        assert(!values.empty() && "expected index value");
        Value result = values.front();
        values = values.drop_front();
        return result;
      }
      return IntegerAttr::get(indexType, d);
    };
    result.sizes.reserve(shape.size());
    result.strides.reserve(strides.size());

    for (int64_t d : shape)
      result.sizes.push_back(handleElement(d));
    for (int64_t d : strides)
      result.strides.push_back(handleElement(d));
    return result;
  }

  ResolvedMetadata resolveMetadata(memref::ExtractStridedMetadataOp op) {
    ResolvedMetadata result;
    result.base = op.getBaseBuffer();
    result.offset = op.getConstifiedMixedOffset();
    result.sizes = op.getConstifiedMixedSizes();
    result.strides = op.getConstifiedMixedStrides();
    return result;
  }

  FailureOr<Value> reconstructMemRef(OpBuilder &b, Location loc,
                                     ValueRange values, MemRefType type) {
    int64_t offset = 0;
    SmallVector<int64_t, 4> strides;
    if (failed(type.getStridesAndOffset(strides, offset)))
      return {};

    size_t expectedSize =
        1 + type.getNumDynamicDims() + (ShapedType::isDynamic(offset) ? 1 : 0) +
        (llvm::count_if(strides,
                        [](int64_t d) { return ShapedType::isDynamic(d); }));

    if (values.size() != expectedSize)
      return failure();

    ResolvedMetadata stridedMetadata =
        resolveMetadata(values, offset, type.getShape(), strides);

    auto result = b.create<memref::ReinterpretCastOp>(
        loc, type, stridedMetadata.base, stridedMetadata.offset,
        stridedMetadata.sizes, stridedMetadata.strides);

    return result.getResult();
  }

  FailureOr<ResolvedMetadata> decomposeMemRef(OpBuilder &b, Location loc,
                                              TypedValue<MemRefType> value) {
    int64_t offset = 0;
    SmallVector<int64_t, 4> strides;
    MemRefType type = value.getType();
    if (failed(type.getStridesAndOffset(strides, offset)))
      return {};
    auto extractOp = b.create<memref::ExtractStridedMetadataOp>(loc, value);
    ResolvedMetadata metadata = resolveMetadata(extractOp);
    return metadata;
  }

private:
  MLIRContext *context;
  Type indexType = IndexType::get(context);
};

/// The following example illustrates the transformation:
///
/// ```
/// func.func @foo(
///  %arg0: memref<4x?xf32, strided<offset: ?, strides: [?, 1]>>) {
///   ...
/// }
/// ```
///
/// becomes
///
/// ```
/// func.func @foo(%arg0_ptr: memref<f32>, %arg0_offset: i64,
///                %arg0_dim1: i64, %arg0_stride0: i64) {
///   %arg0 = memref.reinterpret_cast %arg0_ptr to
///            offset: [%arg0_offset],
///            sizes: [4, %arg0_dim1],
///            strides: [%arg0_stride0, 1]
///            : memref<f32> to memref<4x?xf32, strided<offset: ?, strides:
///            [?, 1]>>
///   ...
/// }
/// ```
///
/// We then use the "bare-pointer" calling convention to lower the function
/// during the lowering to the lower-level target.
class KernelExpandMemRefArgsPass
    : public kernel::impl::KernelExpandMemRefArgsPassBase<
          KernelExpandMemRefArgsPass> {
  using Base::Base;

  FrozenRewritePatternSet gpuModulePatterns;
  std::shared_ptr<MemRefTypeConverter> typeConverter;
  std::shared_ptr<ConversionTarget> gpuTarget;

  LogicalResult initialize(MLIRContext *context) override {
    typeConverter = std::make_shared<MemRefTypeConverter>(context);

    gpuModulePatterns = [&]() {
      RewritePatternSet patterns(context);
      mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, *typeConverter);
      return patterns;
    }();

    gpuTarget = std::make_shared<ConversionTarget>(*context);
    gpuTarget->addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      return typeConverter->isSignatureLegal(func.getFunctionType());
    });
    gpuTarget->markUnknownOpDynamicallyLegal(
        [](Operation *op) { return true; });

    return success();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();
    if (failed(applyPartialConversion(module, *gpuTarget, gpuModulePatterns))) {
      emitError(module.getLoc(),
                "failed to expand memref arguments in GPU kernel");
      return signalPassFailure();
    }
  }
};
} // namespace
