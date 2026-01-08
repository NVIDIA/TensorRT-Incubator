//===- ConversionTarget.cpp  ----------------------------------------------===//
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
/// Implementation of common Executor dialect conversion infrastructure.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-executor/Utils/MemRefDescriptorAdaptor.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::executor;

static Value createIntegerConstant(ImplicitLocOpBuilder &builder,
                                   Type resultType, uint64_t value) {
  return builder.create<executor::ConstantOp>(
      builder.getIntegerAttr(resultType, value));
}

//===----------------------------------------------------------------------===//
// Executor Conversion Target
//===----------------------------------------------------------------------===//

ExecutorConversionTarget::ExecutorConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<executor::ExecutorDialect>();
  addLegalOp<UnrealizedConversionCastOp>();
}

//===----------------------------------------------------------------------===//
// Helpers for checking for or setting the DataLayout specification
// on a module.
//===----------------------------------------------------------------------===//

std::optional<DataLayoutSpecInterface>
executor::getDataLayoutSpec(ModuleOp op) {
  auto spec = op->getAttrOfType<DataLayoutSpecInterface>(
      DLTIDialect::kDataLayoutAttrName);
  if (spec)
    return spec;
  return {};
}

FailureOr<DataLayout> executor::setDataLayoutSpec(Operation *op,
                                                  uint64_t indexBitwidth,
                                                  uint64_t pointerBitWidth) {
  ModuleOp module =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  if (!module)
    return failure();
  std::optional<DataLayoutSpecInterface> spec = getDataLayoutSpec(module);
  MLIRContext *ctx = module->getContext();
  auto i64Attr = [&](int64_t val) {
    return IntegerAttr::get(IntegerType::get(ctx, 64), val);
  };
  auto getPointerType = [&](MemoryType memType) {
    return executor::PointerType::get(ctx, memType);
  };
  auto indexType = IndexType::get(ctx);

  if (!spec) {
    module->setAttr(
        DLTIDialect::kDataLayoutAttrName,
        DataLayoutSpecAttr::get(
            ctx, {DataLayoutEntryAttr::get(indexType, i64Attr(indexBitwidth)),
                  DataLayoutEntryAttr::get(getPointerType(MemoryType::host),
                                           i64Attr(pointerBitWidth)),
                  DataLayoutEntryAttr::get(getPointerType(MemoryType::device),
                                           i64Attr(pointerBitWidth))}));
    return DataLayout(module);
  }

  DataLayout layout(module);
  llvm::TypeSize indexSize = layout.getTypeSizeInBits(indexType);
  if (indexSize.isScalable() || indexSize.getFixedValue() != indexBitwidth)
    return failure();

  for (Type ptr :
       {getPointerType(MemoryType::host), getPointerType(MemoryType::device)}) {
    llvm::TypeSize s = layout.getTypeSizeInBits(ptr);
    if (s.isScalable() || s.getFixedValue() != pointerBitWidth)
      return failure();
  }
  return layout;
}

//===----------------------------------------------------------------------===//
// Executor Type Converter
//===----------------------------------------------------------------------===//

ExecutorTypeConverter::ExecutorTypeConverter(
    MLIRContext *ctx, const LowerToExecutorOptions &options,
    DataLayout dataLayout)
    : options(options),
      dialect(ctx->getOrLoadDialect<executor::ExecutorDialect>()),
      dataLayout(std::move(dataLayout)) {
  assert(dialect && "expected loaded Executor dialect");
  assert((isa<IntegerType>(options.indexType) ||
          isa<IndexType>(options.indexType)) &&
         "expected indexType to be an integer type");

  FloatType f32Type = Float32Type::get(ctx);
  FloatType f64Type = Float64Type::get(ctx);

  // Add conversions for supported POD types.
  addConversion([&](FunctionType t) -> std::optional<Type> {
    SmallVector<Type> convertedInputs, convertedResults;
    if (failed(convertTypes(t.getInputs(), convertedInputs)) ||
        failed(convertTypes(t.getResults(), convertedResults)))
      return std::nullopt;
    return FunctionType::get(t.getContext(), convertedInputs, convertedResults);
  });
  addConversion(
      [indexType = getIndexType()](IndexType t) { return indexType; });
  addConversion([](IntegerType t) -> std::optional<Type> {
    if (t.isSignless())
      return t;
    return IntegerType::get(t.getContext(), t.getWidth());
  });
  addConversion([f32Type, f64Type](ComplexType t) -> std::optional<Type> {
    if (t == ComplexType::get(f32Type) || t == ComplexType::get(f64Type))
      return executor::TableType::get(t.getContext(),
                                      {t.getElementType(), t.getElementType()});
    return {};
  });
  addConversion([](FloatType t) -> std::optional<Type> {
    int64_t bitwidth = t.getWidth();
    if (bitwidth == 32 || bitwidth == 16 || bitwidth == 64 || bitwidth == 8 ||
        bitwidth == 4)
      return t;
    return std::nullopt;
  });

  addConversion([](Type t) -> std::optional<Type> {
    if (isa<executor::PointerType, executor::StrLiteralType>(t))
      return t;
    return std::nullopt;
  });

  // Recursively check table bodies.
  addConversion([&](executor::TableType t) -> std::optional<Type> {
    SmallVector<Type> bodyTypes;
    if (failed(convertTypes(t.getBody(), bodyTypes)))
      return std::nullopt;
    return executor::TableType::get(t.getContext(), bodyTypes);
  });

  // Memrefs are converted to tables.
  addConversion([&](MemRefType t) -> std::optional<Type> {
    FailureOr<SmallVector<Type>> fields = getMemRefDescriptorFields(t);
    if (failed(fields))
      return std::nullopt;
    return executor::TableType::get(t.getContext(), *fields);
  });

  // Fallback source materializer.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  // Memref source materializer.
  addSourceMaterialization([&](OpBuilder &b, MemRefType memrefType,
                               ValueRange components, Location loc) -> Value {
    if (MemRefDescriptor::isMemRefDescriptorFieldTypes(
            memrefType, getIndexType(), TypeRange(components))) {
      ImplicitLocOpBuilder builder(loc, b);
      auto descriptor = MemRefDescriptor::fromComponents(
          builder, *this, memrefType, components[0], components[1],
          components[2], components.slice(3, memrefType.getRank()),
          components.slice(3 + memrefType.getRank(), memrefType.getRank()));
      return b
          .create<UnrealizedConversionCastOp>(loc, memrefType,
                                              Value(descriptor))
          .getResult(0);
    }
    assert(components.size() == 1 &&
           "expected memref to be passed as a single value");
    return b.create<UnrealizedConversionCastOp>(loc, memrefType, components)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc,
                               Type originalType) -> Value {
    if (auto memrefType = dyn_cast_if_present<MemRefType>(originalType)) {
      if (MemRefDescriptor::isMemRefDescriptorFieldTypes(
              memrefType, getIndexType(), TypeRange(inputs))) {
        ImplicitLocOpBuilder b(loc, builder);
        auto descriptor = MemRefDescriptor::fromComponents(
            b, *this, memrefType, inputs[0], inputs[1], inputs[2],
            inputs.slice(3, memrefType.getRank()),
            inputs.slice(3 + memrefType.getRank(), memrefType.getRank()));
        return descriptor;
      }
    }
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTypeAttributeConversion([&](TableType table, TypeAttr typeAttr) {
    if (isa<IndexType>(typeAttr.getValue()))
      return TypeAttr::get(getIndexType());
    return typeAttr;
  });
}

/// Converts a function argument type to an executor type(s) and appends the
/// types to `results`.
static FailureOr<SmallVector<Type>>
convertFuncArg(const ExecutorTypeConverter &converter, Type type) {
  SmallVector<Type> results;
  if (isa<UnrankedMemRefType>(type))
    return failure();
  Type converted = converter.convertType(type);
  if (!converted)
    return failure();
  results.push_back(converted);
  return results;
}

/// Convert the function type composed of `inputs` and `results`. The converted
/// result types are returned and the SignatureConversion contains the
/// converted input types.
static FailureOr<SmallVector<Type>>
convertSignatureImpl(TypeRange inputs, TypeRange results,
                     const ExecutorTypeConverter &typeConverter,
                     ExecutorTypeConverter::SignatureConversion &result) {
  for (auto [it, argTy] : llvm::enumerate(inputs)) {
    FailureOr<SmallVector<Type>> args = convertFuncArg(typeConverter, argTy);
    if (failed(args))
      return failure();
    result.addInputs(it, *args);
  }
  SmallVector<Type> resultTypes;
  for (Type t : results) {
    Type converted = typeConverter.convertType(t);
    if (!converted)
      return failure();
    resultTypes.push_back(converted);
  }
  return resultTypes;
}

Type ExecutorTypeConverter::convertFunctionSignature(
    FunctionType funcType, SignatureConversion &result) const {
  FailureOr<SmallVector<Type>> resultTypes = convertSignatureImpl(
      funcType.getInputs(), funcType.getResults(), *this, result);
  if (failed(resultTypes))
    return {};
  return FunctionType::get(funcType.getContext(), result.getConvertedTypes(),
                           *resultTypes);
}

ExecutorFunctionType ExecutorTypeConverter::convertExecutorFunctionSignature(
    ExecutorFunctionType funcType, SignatureConversion &result) const {
  FailureOr<SmallVector<Type>> resultTypes = convertSignatureImpl(
      funcType.getArgs(), funcType.getResults(), *this, result);
  if (failed(resultTypes))
    return {};
  return ExecutorFunctionType::get(funcType.getContext(),
                                   result.getConvertedTypes(), *resultTypes,
                                   funcType.getTrailingVarArg());
}

FailureOr<SmallVector<Type>> ExecutorTypeConverter::getMemRefDescriptorFields(
    MemRefType type, std::optional<MemoryType> space) const {
  if (type.getMemorySpace() && !isa<MemoryTypeAttr>(type.getMemorySpace()))
    llvm::report_fatal_error(
        "the 'memref-to-executor' type converter does not allow memory "
        "space attributes that are not of #executor.memory_space type");
  MemoryTypeAttr spaceAttr =
      dyn_cast_or_null<MemoryTypeAttr>(type.getMemorySpace());
  MemoryType memorySpace =
      space.has_value() ? *space
                        : (spaceAttr ? spaceAttr.getValue() : MemoryType::host);

  if (!type.isStrided())
    return failure();

  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return failure();

  Type indexTy = getIndexType();
  Type pointerType = getOpaquePointerType(memorySpace);
  int64_t rank = type.getRank();
  SmallVector<Type> results;
  results.reserve(3 + 2 * rank);
  results.append({pointerType, pointerType, indexTy});
  if (rank == 0)
    return results;
  results.append(2 * rank, indexTy);
  return results;
}

Type ExecutorTypeConverter::getIndexType() const { return options.indexType; }

Type ExecutorTypeConverter::getOpaquePointerType(MemoryType type) const {
  return executor::PointerType::get(dialect->getContext(), type);
}

std::string ExecutorTypeConverter::convertOpNameToBackendBuiltinFuncName(
    StringRef opName) const {
  return "_" + llvm::join(llvm::split(opName, "."), "_");
}

uint64_t ExecutorTypeConverter::getMemRefElementTypeByteSize(
    MemRefType memrefType) const {
  Type elType = memrefType.getElementType();
  if (auto subMemRef = dyn_cast<MemRefType>(elType)) {
    assert(!subMemRef.getMemorySpace() && "asssumed no memory space");
    return MemRefDescriptorAdaptor::getDescriptorByteSize(
        subMemRef,
        dataLayout.getTypeSize(getOpaquePointerType(MemoryType::host)),
        dataLayout.getTypeSize(getOptions().indexType));
  }
  return dataLayout.getTypeSize(elType);
}

// Check if a memref type can be converted to a bare pointer.
static bool canConvertToBarePtr(BaseMemRefType type) {
  if (isa<UnrankedMemRefType>(type))
    // Unranked memref is not supported in the bare pointer calling convention.
    return false;

  // Check that the memref has static shape, strides and offset. Otherwise, it
  // cannot be lowered to a bare pointer.
  auto memrefTy = cast<MemRefType>(type);
  if (!memrefTy.hasStaticShape())
    return false;

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(memrefTy.getStridesAndOffset(strides, offset)))
    return false;

  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride))
      return false;

  return !ShapedType::isDynamic(offset);
}

LogicalResult ExecutorTypeConverter::promoteOperands(
    Location loc, ValueRange opOperands, ValueRange operands,
    OpBuilder &builder, bool useBarePtrCallConv,
    SmallVectorImpl<Value> &promotedOperands) const {
  promotedOperands.reserve(operands.size());
  for (auto [operand, llvmOperand] : llvm::zip(opOperands, operands)) {
    if (useBarePtrCallConv) {
      // For the bare-ptr calling convention, we only have to extract the
      // aligned pointer of a memref.
      if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
        if (!canConvertToBarePtr(memrefType))
          return failure();
        MemRefDescriptor desc(llvmOperand, memrefType);
        llvmOperand = desc.alignedPtr(builder, loc);
      } else if (isa<UnrankedMemRefType>(operand.getType())) {
        return failure();
      }
    } else {
      if (isa<UnrankedMemRefType>(operand.getType())) {
        return failure();
      }
      if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
        MemRefDescriptor desc(operand, memrefType);
        llvm::append_range(promotedOperands, desc.unpack(builder, loc));
        continue;
      }
    }
    promotedOperands.push_back(llvmOperand);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Executor Derived Conversion Pattern Rewriters
//===----------------------------------------------------------------------===//

Value ConvertToExecutorPattern::getCUDADeviceId(OpBuilder &builder,
                                                Location loc,
                                                ModuleOp module) const {
  MLIRContext *context = getTypeConverter()->getContext();
  Type i32Type = IntegerType::get(context, 32);
  ExecutorCallBuilder externalCallBuilder = {
      context, "__spmd_global_rank", {i32Type}, {}};
  Value rank =
      externalCallBuilder.create(builder, loc, module, {}).getResult(0);
  return rank;
}

Value ConvertToExecutorPattern::createIndexConstant(ImplicitLocOpBuilder &b,
                                                    int64_t value) const {
  auto indexTy = getTypeConverter()->getIndexType();
  return b.create<executor::ConstantOp>(b.getIntegerAttr(indexTy, value));
}

FailureOr<uint64_t> ConvertToExecutorPattern::getTypeSizeInBytes(Type t) const {
  llvm::TypeSize bytes = getDataLayout().getTypeSize(t);
  if (bytes.isScalable())
    return failure();
  return bytes.getFixedValue();
}

FailureOr<ConvertToExecutorPattern::MemRefAllocationInformation>
ConvertToExecutorPattern::getMemRefAllocationInformation(
    ImplicitLocOpBuilder &b, MemRefType memrefType,
    ValueRange dynamicSizes) const {
  if (!getTypeConverter()->convertType(memrefType) ||
      !memrefType.getLayout().isIdentity())
    return emitError(b.getLoc())
           << "the allocated type of 'memref.alloc' must have the identity "
              "layout for conversion to Executor dialect, but the result type "
              "is "
           << memrefType;

  // Get element byte size.
  Type elementType = typeConverter->convertType(memrefType.getElementType());
  FailureOr<int64_t> elementByteSize = getTypeSizeInBytes(elementType);
  if (failed(elementByteSize))
    return failure();

  assert(static_cast<unsigned>(memrefType.getNumDynamicDims()) ==
             dynamicSizes.size() &&
         "must provide correct dynamic dims");

  MemRefAllocationInformation info;
  info.sizes.reserve(memrefType.getRank());
  unsigned dynamicIndex = 0;
  for (int64_t size : memrefType.getShape())
    info.sizes.push_back(size == ShapedType::kDynamic
                             ? dynamicSizes[dynamicIndex++]
                             : createIndexConstant(b, size));
  info.memorySpace = MemoryType::host;
  if (Attribute memorySpaceAttr = memrefType.getMemorySpace()) {
    auto memoryTypeAttr = dyn_cast<MemoryTypeAttr>(memorySpaceAttr);
    if (!memoryTypeAttr)
      return emitError(b.getLoc())
             << "memref types must only have address "
                "spaces of type MemoryTypeAttr for conversion to Executor IR";
    info.memorySpace = memoryTypeAttr.getValue();
  }

  // Since we enforce having the identity type, calculate the stride using
  // suffix product.
  Value runningStride = createIndexConstant(b, 1);
  info.strides.resize(memrefType.getRank());
  for (int64_t i = memrefType.getRank() - 1; i >= 0; i--) {
    info.strides[i] = runningStride;
    runningStride = b.create<executor::MulIOp>(runningStride, info.sizes[i]);
  }

  // Perform ceiling-divide in case we have some bitwidth that is not a multiple
  // of 8.
  info.sizeBytes = b.create<executor::MulIOp>(
      runningStride,
      createIntegerConstant(b, runningStride.getType(), *elementByteSize));
  return info;
}

Value ConvertToExecutorPattern::getLinearizedOffset(
    ImplicitLocOpBuilder &b, const MemRefDescriptor &descriptor,
    ValueRange indices) const {
  auto memrefType = descriptor.getMemRefType();
  auto [strides, offset] = memrefType.getStridesAndOffset();
  Value result = ShapedType::isDynamic(offset) ? descriptor.offset(b)
                                               : createIndexConstant(b, offset);
  assert(memrefType.getRank() == static_cast<int64_t>(indices.size()) &&
         strides.size() == indices.size() &&
         "expected equal strides and indices");
  unsigned idx = 0;
  for (auto [index, stride] : llvm::zip(indices, strides)) {
    Value increment = index;
    if (stride != 1 && !matchPattern(increment, m_Zero())) {
      Value stride = ShapedType::isDynamic(strides[idx])
                         ? descriptor.stride(b, idx)
                         : createIndexConstant(b, strides[idx]);
      increment = b.create<executor::MulIOp>(increment, stride);
    }
    result = result ? b.create<executor::AddIOp>(result, increment) : increment;
    idx++;
  }
  return result;
}

Value ConvertToExecutorPattern::convertOffsetInElementsToBytes(
    ImplicitLocOpBuilder &b, Value offsetInElements,
    MemRefType memRefType) const {
  const ExecutorTypeConverter *typeConverter = this->getTypeConverter();
  return b.create<executor::GetOffsetOp>(
      typeConverter->getIndexType(),
      typeConverter->convertType(memRefType.getElementType()),
      ArrayRef<OpFoldResult>(offsetInElements));
}

PointerType ConvertToExecutorPattern::getHostPointerType() const {
  return PointerType::get(getContext(), MemoryType::host);
}

PointerType ConvertToExecutorPattern::getHostPinnedPointerType() const {
  return PointerType::get(getContext(), MemoryType::host_pinned);
}

PointerType ConvertToExecutorPattern::getDevicePointerType() const {
  return PointerType::get(getContext(), MemoryType::device);
}

SmallVector<Value> ConvertToExecutorPattern::convertFuncCallOperands(
    RewriterBase &rewriter, Location loc, ValueRange originalOperands,
    ValueRange adaptorOperands) const {
  ImplicitLocOpBuilder b(loc, rewriter);
  SmallVector<Value> operands;
  for (auto [original, converted] :
       llvm::zip_equal(originalOperands, adaptorOperands))
    operands.push_back(converted);
  return operands;
}

/// Returns `true` if a memref with shape `shape` and `strides` represents a
/// contiguous array of memory. This is equivalent to checking whether some
/// subview is contiguous. The idea here is that the shape and laout should have
/// a canonical row-major layout when removing the unit extents. For example,
/// `memref<8x1x4, strided<[4, 32, 1], offset: ?>>` should be contiguous since
/// we can ignore the middle unit extent dimension.
static bool isContiguousImpl(ArrayRef<int64_t> strides,
                             ArrayRef<int64_t> shape) {
  unsigned e = strides.size();
  if (shape.empty() || strides.empty())
    return true;

  auto findNextIndex = [&](unsigned start) -> std::optional<unsigned> {
    for (unsigned i = start; i < e; i++) {
      if (shape[i] != 1)
        return i;
    }
    return {};
  };

  // If no starting index, then this is a scalar shape.
  std::optional<unsigned> index = findNextIndex(0);
  if (!index)
    return true;

  while (*index < e) {
    std::optional<unsigned> next = findNextIndex(*index + 1);
    // If this is the last relevant index, it must be unit stride or unit
    // access.
    if (!next)
      return strides[*index] == 1 || shape[*index] == 1;
    if (ShapedType::isDynamic(strides[*index]) ||
        ShapedType::isDynamic(strides[*next]))
      return false;
    if (strides[*index] != strides[*next] * shape[*next])
      return false;
    index = *next;
  }
  return true;
}

/// Returns `true` if `t` represents a contiuous area of memory. This is true if
/// (1) `t` has canonical strides (identity layout) or (2) `t` has non-identity
/// strides but is size `1` in all dimensions where the stride is non-canonical.
bool ConvertToExecutorPattern::isContiguous(MemRefType t) {
  if (t.getLayout().isIdentity())
    return true;
  if (!t.hasStaticShape())
    return false;
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(t.getStridesAndOffset(strides, offset)))
    return false;
  return isContiguousImpl(strides, t.getShape());
}

/// Returns `true` if a copy from `srcMemRefType` to `dstMemRefType` requires a
/// strided copy, `false` if it can be expressed as a single memcpy. If it can
/// be, return the offset for the destination memref, otherwise return failure.
bool ConvertToExecutorPattern::isCopyStrided(MemRefType srcMemRefType,
                                             MemRefType dstMemRefType) {
  return !isContiguous(srcMemRefType) || !isContiguous(dstMemRefType);
}

std::optional<MemoryType>
ConvertToExecutorPattern::getMemorySpace(MemRefType type) {
  auto srcMemoryTypeAttr =
      dyn_cast_or_null<MemoryTypeAttr>(type.getMemorySpace());
  if (!srcMemoryTypeAttr)
    return MemoryType::host;
  return srcMemoryTypeAttr.getValue();
}

bool ConvertToExecutorPattern::isHostVisibleOnlyMemoryType(MemRefType type) {
  auto space = getMemorySpace(type);
  if (!space)
    return false;
  return *space == MemoryType::host || *space == MemoryType::host_pinned;
}

bool ConvertToExecutorPattern::isDeviceVisibleMemoryType(MemRefType type) {
  auto space = getMemorySpace(type);
  if (!space)
    return false;
  return *space == MemoryType::device || *space == MemoryType::unified;
}

bool ConvertToExecutorPattern::isHostVisibleMemoryType(MemRefType type) {
  auto space = getMemorySpace(type);
  if (!space)
    return false;
  return *space == MemoryType::host || *space == MemoryType::host_pinned ||
         *space == MemoryType::unified;
}

//===----------------------------------------------------------------------===//
// ExecutorMemRefBuilder
//===----------------------------------------------------------------------===//

static Value buildExtractValue(OpBuilder &b, Location loc, Value agg,
                               unsigned pos) {
  return b.create<executor::ExtractTableValueOp>(loc, agg, pos);
}
static Value buildInsertValue(OpBuilder &b, Location loc, Value agg, Value item,
                              unsigned pos) {
  return b.create<executor::InsertTableValueOp>(loc, agg, item, pos);
}
static Value buildConstantIndex(OpBuilder &b, Location loc, Type indexType,
                                int64_t value) {
  return b.create<executor::ConstantOp>(loc,
                                        b.getIntegerAttr(indexType, value));
}

/// Construct a helper for the given descriptor value.
MemRefDescriptor::MemRefDescriptor(Value descriptor, MemRefType memrefType)
    : MemRefDescriptorAdaptor(descriptor, memrefType, buildExtractValue,
                              buildInsertValue, buildConstantIndex,
                              cast<executor::TableType>(descriptor.getType())
                                  .getBody()[kOffsetPosInMemRefDescriptor]),
      indexType(cast<executor::TableType>(descriptor.getType())
                    .getBody()[kOffsetPosInMemRefDescriptor]) {
  assert(value != nullptr && "value cannot be null");
}

MemRefDescriptor MemRefDescriptor::fromComponents(
    ImplicitLocOpBuilder &b, const ExecutorTypeConverter &typeConverter,
    MemRefType type, Value allocatedPtr, Value alignedPtr, Value offset,
    ValueRange sizes, ValueRange strides) {
  Type descriptorType = typeConverter.convertType(type);
  Value descriptor = b.create<executor::CreateTableOp>(
      descriptorType, allocatedPtr, alignedPtr, offset, sizes, strides);
  return MemRefDescriptor(descriptor, type);
}

MemRefDescriptor MemRefDescriptor::fromComponents(
    ImplicitLocOpBuilder &b, const ExecutorTypeConverter &typeConverter,
    MemRefType type, Value allocatedPtr, Value alignedPtr, OpFoldResult offset,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  Type indexType = typeConverter.getIndexType();
  Type descriptorType = typeConverter.convertType(type);
  auto correctAttr = [&](OpFoldResult ofr) -> OpFoldResult {
    if (isa<Value>(ofr))
      return ofr;
    auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
    if (attr.getType() == indexType)
      return cast<Attribute>(ofr);
    return b.getIntegerAttr(indexType, attr.getInt());
  };
  auto correctAttrs = [&](ArrayRef<OpFoldResult> ofrs) {
    return llvm::to_vector(llvm::map_range(ofrs, correctAttr));
  };
  Value descriptor = b.create<executor::CreateTableOp>(
      descriptorType, allocatedPtr, alignedPtr, correctAttr(offset),
      correctAttrs(sizes), correctAttrs(strides));
  return MemRefDescriptor(descriptor, type);
}

void MemRefDescriptor::setConstantOffset(ImplicitLocOpBuilder &b,
                                         uint64_t offset) {
  setOffset(b, createIntegerConstant(b, indexType, offset));
}

void MemRefDescriptor::setConstantSize(ImplicitLocOpBuilder &b, unsigned pos,
                                       uint64_t size) {
  setSize(b, pos, createIntegerConstant(b, indexType, size));
}

void MemRefDescriptor::setConstantStride(ImplicitLocOpBuilder &b, unsigned pos,
                                         uint64_t stride) {
  setStride(b, pos, createIntegerConstant(b, indexType, stride));
}

SmallVector<Value> MemRefDescriptor::sizes(OpBuilder &b, Location loc) const {
  SmallVector<Value> sizes;
  for (int64_t i = 0; i < getMemRefType().getRank(); i++) {
    if (memrefType.isDynamicDim(i))
      sizes.push_back(size(b, loc, i));
    else
      sizes.push_back(b.create<executor::ConstantOp>(
          loc, indexType,
          b.getIntegerAttr(indexType, memrefType.getDimSize(i))));
  }
  return sizes;
}

SmallVector<Value> MemRefDescriptor::strides(OpBuilder &b, Location loc) const {
  SmallVector<Value> result;
  auto [strides, offset] =
      const_cast<MemRefType &>(memrefType).getStridesAndOffset();
  for (int64_t i = 0; i < getMemRefType().getRank(); i++) {
    if (ShapedType::isDynamic(strides[i]))
      result.push_back(stride(b, loc, i));
    else
      result.push_back(b.create<executor::ConstantOp>(
          loc, indexType, b.getIntegerAttr(indexType, strides[i])));
  }
  return result;
}

Value MemRefDescriptor::shapeVolumeInElements(ImplicitLocOpBuilder &b) const {
  if (getMemRefType().getRank() == 0)
    return b.create<executor::ConstantOp>(b.getIntegerAttr(indexType, 1));

  if (memrefType.hasStaticShape())
    return b.create<executor::ConstantOp>(
        b.getIntegerAttr(indexType, memrefType.getNumElements()));

  // Compute dynamic product.
  Value numElements =
      b.create<executor::ConstantOp>(b.getIntegerAttr(indexType, 1));
  for (int pos = 0; pos < memrefType.getRank(); pos++)
    numElements = b.create<executor::MulIOp>(numElements, this->size(b, pos));
  return numElements;
}

Value MemRefDescriptor::shapeVolumeInBytes(ImplicitLocOpBuilder &b) const {
  auto elementType = getMemRefType().getElementType();
  Value numElements = shapeVolumeInElements(b);
  return b.create<executor::MulIOp>(
      numElements,
      b.create<executor::ConstantOp>(b.getIntegerAttr(
          indexType,
          llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8))));
}

bool MemRefDescriptor::isMemRefDescriptorFieldTypes(MemRefType originalType,
                                                    Type indexType,
                                                    TypeRange types) {
  if (static_cast<int64_t>(types.size()) != 3 + 2 * originalType.getRank())
    return false;
  return llvm::all_of(types.take_front(2),
                      llvm::IsaPred<executor::PointerType>) &&
         llvm::all_of(types.drop_front(2),
                      [&](Type t) { return t == indexType; });
}

//===----------------------------------------------------------------------===//
// ExecutorCallBuilder
//===----------------------------------------------------------------------===//

ExecutorCallBuilder::ExecutorCallBuilder(MLIRContext *ctx,
                                         StringRef functionName,
                                         ArrayRef<Type> returnType,
                                         ArrayRef<Type> argumentTypes,
                                         bool trailingVarArgs)
    : functionName(functionName),
      functionType(executor::ExecutorFunctionType::get(
          ctx, argumentTypes, returnType,
          trailingVarArgs ? UnitAttr::get(ctx) : nullptr)) {}

executor::CallOp ExecutorCallBuilder::create(OpBuilder &builder, Location loc,
                                             ModuleOp module,
                                             ValueRange arguments) const {
  auto *context = module.getContext();
  SymbolRefAttr callee = SymbolRefAttr::get(context, functionName);
  if (!module.lookupSymbol<FuncOp>(functionName)) {
    // Insert the private function declaration into the body of the parent
    // module.
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcOp = builder.create<FuncOp>(loc, functionName, functionType);
    funcOp.setSymVisibility("private");
  }
  return builder.create<executor::CallOp>(loc, functionType.getResults(),
                                          callee.getLeafReference(), arguments);
}
