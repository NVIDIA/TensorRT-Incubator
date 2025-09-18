//===- TranslateToRuntimeExecutable.cpp -----------------------------------===//
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
/// Definitions for translation `mlir-to-runtime-executable`.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir-executor/Target/Lua/TranslateToLua.h"
#include "mlir-executor/Utils/SerializationUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {
namespace fb = ::flatbuffers;

/// Aliases flatbuffers 32bit offset.
template <typename T>
using Offset = fb::Offset<T>;

/// Aliases flatbuffers 64bit offset.
template <typename T>
using Offset64 = fb::Offset64<T>;

/// Describes a 64bit offset pair.
template <typename T1, typename T2>
using Offset64Pair = std::pair<Offset64<T1>, Offset64<T2>>;

/// Describes a 32bit offset for an item serialized as a union.
template <typename T1>
using UnionOffset = std::pair<T1, Offset<void>>;

/// A wrapper around `flatbuffers::FlatBufferBuilder64` that provides a
/// convenient interface for serializing data into the flatbuffer.
///
/// The `serialize64` methods should be used for serializing data
/// with 64-bit offsets into the buffer.
///
/// The `serialize` methods should be used for serializing data with 32-bit
/// offsets into the buffer.
///
/// All 64-bit offset data must be serialized prior to the 32-bit offset data.
class FBBuilder : public fb::FlatBufferBuilder64 {
public:
  /// Serialize a vector of elements into the buffer.
  template <typename T>
  auto serialize(const std::vector<T> &span) {
    return this->CreateVector(span);
  }

  /// Serialize a span of elements into the buffer.
  template <typename T>
  auto serialize(mlir::ArrayRef<T> span) {
    return this->CreateVector(span.data(), span.size());
  }

  /// Serialize a small vector of elements into the buffer.
  template <typename T>
  auto serialize(mlir::SmallVector<T> span) {
    return this->serialize(ArrayRef<T>(span));
  }

  /// Serialize a span of elements into the buffer as a 64bit vector.
  template <typename T>
  auto serialize64(mlir::ArrayRef<T> span) {
    return this->CreateVector<T, fb::Offset64, fb::Vector64>(span.data(),
                                                             span.size());
  }

  /// Serialize an ElementsAttr into the buffer.
  template <typename T, template <typename...> class OffsetT = fb::Offset64,
            template <typename...> class VectorT = fb::Vector64>
  FailureOr<OffsetT<VectorT<T>>>
  serialize64(Location loc, const DataLayout &dataLayout, ElementsAttr attr,
              std::optional<uint32_t> alignment = {});

private:
  /// An implementation of `mlir::SerializationInterface` that serializes
  /// elements attributes into the flatbuffer. It is meant to be single use via
  /// the public `serialize` method for `ElementsAttr` above.
  template <typename T, template <typename...> class OffsetT = fb::Offset64,
            template <typename...> class VectorT = fb::Vector64>
  class FlatbufferElementsSerializer : public mlir::SerializationInterface {
  public:
    FlatbufferElementsSerializer(FBBuilder &fb, const DataLayout &dataLayout)
        : mlir::SerializationInterface(dataLayout), fb(fb) {}

    LogicalResult serialize(const char *data, size_t size, Type elementType,
                            uint64_t align) override {
      fb.ForceVectorAlignment64(size, dataLayout.getTypeSize(elementType),
                                align);
      offset = fb.CreateVector<T, OffsetT, VectorT>(
          reinterpret_cast<const T *>(data), size);
      return success();
    }

    OffsetT<VectorT<T>> getOffset() const { return offset; }

  private:
    FBBuilder &fb;
    OffsetT<VectorT<T>> offset{0};
  };
};

} // namespace

static bool isElidedResourceElementsAttr(ElementsAttr attr) {
  auto denseResourceAttr = dyn_cast<DenseResourceElementsAttr>(attr);
  if (!denseResourceAttr)
    return false;
  DenseResourceElementsHandle handle = denseResourceAttr.getRawHandle();
  if (handle.getKey() != "__elided__")
    return false;
  return true;
}

static FailureOr<DenseElementsAttr>
getDenseElementsAttrOfOnes(ElementsAttr attr) {
  ShapedType tensorType = cast<ShapedType>(attr.getType());
  Type elementType = tensorType.getElementType();
  if (elementType.isInteger(1))
    return DenseElementsAttr::get(tensorType, true);
  if (elementType.isInteger(8))
    return DenseElementsAttr::get(tensorType, APInt(8, 1));
  if (elementType.isInteger(16))
    return DenseElementsAttr::get(tensorType, APInt(16, 1));
  if (elementType.isInteger(32))
    return DenseElementsAttr::get(tensorType, APInt(32, 1));
  if (elementType.isInteger(64))
    return DenseElementsAttr::get(tensorType, APInt(64, 1));
  if (isa<Float8E4M3FNType>(elementType))
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::Float8E4M3FN()));
  if (isa<Float4E2M1FNType>(elementType))
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::Float4E2M1FN()));
  if (elementType.isF16())
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::IEEEhalf()));
  if (elementType.isBF16())
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::BFloat()));
  if (elementType.isF32())
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::IEEEsingle()));
  if (elementType.isF64())
    return DenseElementsAttr::get(tensorType,
                                  APFloat::getOne(APFloat::IEEEdouble()));
  if (elementType ==
      ComplexType::get(Float32Type::get(elementType.getContext()))) {
    std::complex<float> complexOne(1.0f, 1.0f);
    return DenseElementsAttr::get(tensorType, complexOne);
  }
  if (elementType ==
      ComplexType::get(Float64Type::get(elementType.getContext()))) {
    std::complex<double> complexOne(1.0, 1.0);
    return DenseElementsAttr::get(tensorType, complexOne);
  }
  return failure();
}

template <typename T, template <typename...> class OffsetT,
          template <typename...> class VectorT>
FailureOr<OffsetT<VectorT<T>>>
FBBuilder::serialize64(Location loc, const DataLayout &dataLayout,
                       ElementsAttr attr, std::optional<uint32_t> alignment) {
  FlatbufferElementsSerializer<T, OffsetT, VectorT> serializer(*this,
                                                               dataLayout);
  if (isElidedResourceElementsAttr(attr)) {
    // Elided attribute can't be serialized so we create splat
    // of `1`s (splat of `true` in case of boolean).
    auto attrOfOnes = getDenseElementsAttrOfOnes(attr);
    if (failed(attrOfOnes))
      return failure();
    attr = *attrOfOnes;
  }
  if (failed(mlir::serializeElementsAttr(loc, attr, dataLayout, serializer,
                                         alignment)))
    return failure();
  return serializer.getOffset();
}

/// Translate the scalar type into the equivalent flatbuffer API object.
static FailureOr<mtrt::ScalarTypeCode> translateScalarType(Type t) {
  if (t.isInteger(32))
    return mtrt::ScalarTypeCode::i32;
  if (t.isInteger(16))
    return mtrt::ScalarTypeCode::i16;
  if (t.isInteger(64))
    return mtrt::ScalarTypeCode::i64;
  if (t.isInteger(8))
    return mtrt::ScalarTypeCode::i8;
  if (t.isF32())
    return mtrt::ScalarTypeCode::f32;
  if (t.isF64())
    return mtrt::ScalarTypeCode::f64;
  if (t.isF16())
    return mtrt::ScalarTypeCode::f16;
  if (t.isBF16())
    return mtrt::ScalarTypeCode::bf16;
  if (isa<Float8E4M3FNType>(t))
    return mtrt::ScalarTypeCode::f8e4m3fn;
  if (isa<Float4E2M1FNType>(t))
    return mtrt::ScalarTypeCode::f4e2m1fn;
  if (t.isIndex())
    return mtrt::ScalarTypeCode::i32;
  if (t.isInteger(1))
    return mtrt::ScalarTypeCode::i1;
  if (t.isInteger(4))
    return mtrt::ScalarTypeCode::i4;
  if (t == ComplexType::get(Float32Type::get(t.getContext())))
    return mtrt::ScalarTypeCode::complex32;
  if (t == ComplexType::get(Float64Type::get(t.getContext())))
    return mtrt::ScalarTypeCode::complex64;
  return failure();
}

/// Serialize the given attribute into the flatbuffer as a Union object. This
/// returns two offsets (one for Bounds enum code, another for the actual
/// concrete bounds object)
static FailureOr<UnionOffset<mtrt::flat::Bounds>>
translateAttribute(FBBuilder &fb, Attribute attr) {
  if (!isa<executor::DimensionBoundsAttr, executor::ValueBoundsAttr, UnitAttr>(
          attr))
    return emitError(UnknownLoc::get(attr.getContext()))
           << "unhandled attribute (" << attr
           << ") in Executor function metadata";

  if (auto dims = llvm::dyn_cast<executor::DimensionBoundsAttr>(attr)) {
    auto min = fb.serialize<int64_t>(dims.getMin());
    auto max = fb.serialize<int64_t>(dims.getMax());
    return std::make_pair(
        mtrt::flat::Bounds::DimensionBounds,
        mtrt::flat::CreateDimensionBounds(fb, min, max).Union());
  }

  if (auto vals = llvm::dyn_cast<executor::ValueBoundsAttr>(attr)) {
    auto toI64 = [](const llvm::APInt &v) { return v.getSExtValue(); };
    auto min = fb.serialize<int64_t>(
        llvm::map_to_vector(vals.getMin().getValues<APInt>(), toI64));
    auto max = fb.serialize<int64_t>(
        llvm::map_to_vector(vals.getMax().getValues<APInt>(), toI64));
    return std::make_pair(mtrt::flat::Bounds::ValueBounds,
                          mtrt::flat::CreateValueBounds(fb, min, max).Union());
  }

  assert(isa<UnitAttr>(attr) && "Must be a unit attribute");
  return std::make_pair(mtrt::flat::Bounds::NoneBounds,
                        mtrt::flat::CreateNoneBounds(fb).Union());
}

/// Translate the memory type into the equivalent flatbuffer API object.
static FailureOr<mtrt::PointerType>
translateMemoryType(executor::MemoryType t) {
  switch (t) {
  case executor::MemoryType::host:
    return mtrt::PointerType::host;
  case executor::MemoryType::host_pinned:
    return mtrt::PointerType::pinned_host;
  case executor::MemoryType::device:
    return mtrt::PointerType::device;
  case executor::MemoryType::unified:
    return mtrt::PointerType::unified;
  default:
    return failure();
  }
}

/// Serialize the given type into the flatbuffer as a Union object. This returns
/// two offsets (one for Type enum code, another for the actual concrete type
/// object.)
static FailureOr<UnionOffset<mtrt::flat::Type>>
translateTypeVariant(FBBuilder &fbBuilder, Type t) {
  auto emitTranslateFailure = [&](Type t) {
    return emitError(UnknownLoc::get(t.getContext()))
           << "unhandled type (" << t << ") in Executor function metadata";
  };

  if (!isa<MemRefType, IntegerType, FloatType, Float8E4M3FNType,
           Float4E2M1FNType, ComplexType>(t))
    return emitTranslateFailure(t);

  // Encode as a memref.
  if (auto memrefType = llvm::dyn_cast<MemRefType>(t)) {
    FailureOr<mtrt::ScalarTypeCode> code =
        translateScalarType(memrefType.getElementType());
    if (failed(code))
      return emitTranslateFailure(memrefType.getElementType());
    auto shape = fbBuilder.serialize<int64_t>(memrefType.getShape());
    auto [strides, offset] = memrefType.getStridesAndOffset();
    auto stridesOffset = fbBuilder.serialize<int64_t>(strides);

    auto addressSpace = mtrt::PointerType::unknown;
    if (llvm::isa_and_nonnull<executor::MemoryTypeAttr>(
            memrefType.getMemorySpace())) {
      auto memoryType =
          executor::PointerType::get(memrefType.getContext(),
                                     llvm::dyn_cast<executor::MemoryTypeAttr>(
                                         memrefType.getMemorySpace())
                                         .getValue())
              .getAddressSpace();
      addressSpace = *translateMemoryType(memoryType);
    }
    return std::make_pair(mtrt::flat::Type::MemRefType,
                          mtrt::flat::CreateMemRefType(fbBuilder, *code, shape,
                                                       stridesOffset,
                                                       addressSpace)
                              .Union());
  }
  // Encode as a scalar type.
  FailureOr<mtrt::ScalarTypeCode> code = translateScalarType(t);
  if (failed(code))
    return emitTranslateFailure(t);
  return std::make_pair(mtrt::flat::Type::ScalarType,
                        mtrt::flat::CreateScalarType(fbBuilder, *code).Union());
}

/// Translate the calling convention.
static mtrt::flat::CallingConvention
translateCallingConvention(executor::CallingConvention cconv) {
  switch (cconv) {
  case executor::CallingConvention::packed:
    return mtrt::flat::CallingConvention::packed;
  case executor::CallingConvention::unpacked:
    return mtrt::flat::CallingConvention::unpacked;
  }
  llvm_unreachable(
      "unknown MLIR Executor -> MTRT runtime calling convention translation");
}

/// Encode the FunctionSignature into the flatbuffer and return the offset of
/// the serialized data.
static FailureOr<Offset<mtrt::flat::FunctionSignature>>
translateSignature(FBBuilder &fbBuilder,
                   executor::FunctionMetadataAttr metadata) {
  assert(metadata && "expected valid FunctionMetadataAttr");

  // Union type must be encoded as variant type + data.
  SmallVector<mtrt::flat::Type> argVariantTypes, resultVariantTypes;
  SmallVector<Offset<void>> argOffsets, resultOffsets;
  argVariantTypes.reserve(metadata.getArgs().size());
  argOffsets.reserve(metadata.getArgs().size());
  resultVariantTypes.reserve(metadata.getResults().size());
  resultOffsets.reserve(metadata.getResults().size());

  for (Type t : metadata.getArgs()) {
    auto arg = translateTypeVariant(fbBuilder, t);
    if (failed(arg))
      return failure();
    auto [argType, argOffset] = *arg;
    argOffsets.push_back(argOffset);
    argVariantTypes.push_back(argType);
  }

  for (Type t : metadata.getResults()) {
    auto res = translateTypeVariant(fbBuilder, t);
    if (failed(res))
      return failure();
    auto [resultType, resultOffset] = *res;
    resultVariantTypes.push_back(resultType);
    resultOffsets.push_back(resultOffset);
  }

  SmallVector<mtrt::flat::Bounds> argBounds, resBounds;
  SmallVector<Offset<void>> argBoundsOffsets, resBoundsOffsets;

  for (Attribute a : metadata.getArgBounds()) {
    auto arg = translateAttribute(fbBuilder, a);
    if (failed(arg))
      return failure();
    auto [argAttr, argOffset] = *arg;
    argBounds.push_back(argAttr);
    argBoundsOffsets.push_back(argOffset);
  }

  for (Attribute a : metadata.getResultBounds()) {
    auto res = translateAttribute(fbBuilder, a);
    if (failed(res))
      return failure();
    auto [resAttr, resOffset] = *res;
    resBounds.push_back(resAttr);
    resBoundsOffsets.push_back(resOffset);
  }

  auto fbBounds = fbBuilder.serialize(argBounds);
  auto fbBoundsOffsets = fbBuilder.serialize(argBoundsOffsets);

  auto shapeFuncSym =
      metadata.getShapeFunc()
          ? fbBuilder.CreateString(metadata.getShapeFunc().getAttr().str())
          : fbBuilder.CreateString("");

  return mtrt::flat::CreateFunctionSignature(
      fbBuilder, fbBuilder.serialize(argVariantTypes),
      fbBuilder.serialize(argOffsets), fbBuilder.serialize(resultVariantTypes),
      fbBuilder.serialize(resultOffsets), metadata.getNumOutputArgs(), fbBounds,
      fbBoundsOffsets, fbBuilder.serialize(resBounds),
      fbBuilder.serialize(resBoundsOffsets), shapeFuncSym,
      translateCallingConvention(metadata.getCconv()));
}

/// Generate a function signature. This is used if there is no explicit
/// 'executor.function_metadata' attached to the function.
static FailureOr<Offset<mtrt::flat::FunctionSignature>>
generateSignature(FBBuilder &fbBuilder, FunctionType metadata) {
  // Union type must be encoded as variant type + data.
  SmallVector<mtrt::flat::Type> argVariantTypes, resultVariantTypes;
  SmallVector<Offset<void>> argOffsets, resultOffsets;
  argVariantTypes.reserve(metadata.getNumInputs());
  argOffsets.reserve(metadata.getNumInputs());
  resultVariantTypes.reserve(metadata.getNumResults());
  resultOffsets.reserve(metadata.getNumResults());
  return mtrt::flat::CreateFunctionSignature(
      fbBuilder, fbBuilder.serialize(argVariantTypes),
      fbBuilder.serialize(argOffsets), fbBuilder.serialize(resultVariantTypes),
      fbBuilder.serialize(resultOffsets), /*num_output_args=*/0,
      /*arg_bounds_type=*/0, /*arg_bounds=*/0, /*result_bounds_type=*/0,
      /*result_bounds=*/0, /*shape_function_name=*/0,
      mtrt::CallingConvention::packed);
}

/// Return a sanitized version of a symbol name by replacing special characters
/// with underscores.
static std::string sanitizeName(StringRef name) {
  SmallVector<StringRef> segments;
  llvm::SplitString(name, segments, "<>\t\n\v\f\r;.!@#$%^&*()-+=");
  return llvm::join(segments, "_");
}

namespace {
/// An implementation of `ExecutableStorage` that just uses a
/// `flatbuffers::DetachedBuffer`. This allows us to avoid a copy of the
/// serialized buffer.
class ExecutableStorageFlatbuffer : public mtrt::ExecutableStorage {
public:
  ExecutableStorageFlatbuffer(flatbuffers::DetachedBuffer storage)
      : storage(std::move(storage)) {}

  const void *data() const final { return storage.data(); }
  size_t size() const final { return storage.size(); }

  std::unique_ptr<ExecutableStorage> getCopy() const final { return nullptr; }

private:
  flatbuffers::DetachedBuffer storage;
};

} // namespace

FailureOr<std::unique_ptr<mtrt::ExecutableStorage>>
mlir::translateToRuntimeExecutable(Operation *op) {

  FBBuilder fbBuilder;
  auto dataLayout = DataLayout::closest(op);

  if (!op->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
      !op->hasTrait<OpTrait::SymbolTable>() || op->getNumRegions() != 1 ||
      !op->getRegion(0).hasOneBlock())
    return emitError(op->getLoc()) << "expected module-like operation";

  // Do rename of symbols to sanitize.
  SymbolTable symbolTable(op);
  for (Operation &op : op->getRegion(0).getOps()) {
    auto nameAttr =
        op.getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
    if (!nameAttr)
      continue;

    std::string sanitized = sanitizeName(nameAttr.strref());
    if (sanitized == nameAttr.strref())
      continue;
    if (failed(symbolTable.rename(&op, sanitized)))
      return emitError(op.getLoc()) << "failed to rename symbol "
                                    << op.getName() << " to sanitized version";
  }

  //===----------------------------------------------------------------------===//
  // 64 bit section
  //===----------------------------------------------------------------------===//

  // For each `executor.data_segment` operation, if there is a constant
  // data value attached to it, then serialize that constant data in the
  // executable as a Constant. These go into the 64bit section. We serialize the
  // string with the data in the 64 bit section.
  SmallVector<Offset64<fb::Vector64<int8_t>>> constData;
  SmallVector<Offset64<fb::String>> globalNames;
  SmallVector<executor::DataSegmentOp> globalOps =
      llvm::to_vector(op->getRegion(0).getOps<executor::DataSegmentOp>());

  for (auto resourceOp : globalOps) {
    if (!resourceOp.getUninitialized()) {
      FailureOr<Offset64<fb::Vector64<int8_t>>> serializedAttr =
          fbBuilder.serialize64<int8_t>(resourceOp.getLoc(), dataLayout,
                                        resourceOp.getValueAttr());
      if (failed(serializedAttr))
        return resourceOp->emitOpError("failed to encode constant value " +
                                       Twine(resourceOp.getSymName()) +
                                       " as a SerializedConstant");
      constData.push_back(*serializedAttr);
    } else {
      constData.push_back(0);
    }

    StringRef name = resourceOp.getSymName();
    globalNames.push_back(
        fbBuilder.CreateString<Offset64>(name.data(), name.size()));
  }

  //===----------------------------------------------------------------------===//
  // 32 bit section
  //===----------------------------------------------------------------------===//
  SmallVector<Offset<mtrt::flat::DataSegment>> constantOffsets;
  constantOffsets.reserve(constData.size());
  for (const auto &[dataOffset, nameOffset, globalOp] :
       llvm::zip_equal(constData, globalNames, globalOps)) {
    FailureOr<uint64_t> uninitializedSize =
        globalOp.getUninitialized()
            ? getSerializedSize(globalOp.getLoc(), globalOp.getValueAttr(),
                                dataLayout)
            : 0;
    if (failed(uninitializedSize))
      return emitError(globalOp.getLoc())
             << "failed to calculate data size for global "
             << globalOp.getSymName();

    FailureOr<mtrt::PointerType> addrSpace =
        translateMemoryType(globalOp.getAddressSpace());
    if (failed(addrSpace))
      return emitError(globalOp.getLoc())
             << "failed to translate address space for global "
             << globalOp.getSymName();
    uint64_t align = dataLayout.getTypeABIAlignment(
        globalOp.getValueAttr().getElementType());
    if (std::optional<uint64_t> alignment = globalOp.getAlignment())
      align = std::max<uint64_t>(align, *alignment);
    constantOffsets.push_back(
        mtrt::flat::CreateDataSegment(fbBuilder, nameOffset, dataOffset,
                                      /*alignment=*/
                                      align,
                                      /*constant=*/globalOp.getConstant(),
                                      /*uninitialized_size=*/*uninitializedSize,
                                      /*address_space=*/*addrSpace));
  }

  std::string sourceString;
  {
    llvm::raw_string_ostream ss(sourceString);
    if (failed(mlir::translateToLua(op, ss)))
      return emitError(op->getLoc(), "Lua translation failed");
  }
  Offset<fb::String> sourceStrOffset = fbBuilder.CreateString(sourceString);

  // Loop over all functions and collect metadata (function names and
  // signatures) that we will embed in the executable.
  SmallVector<Offset<mtrt::flat::Function>> funcOffsets;
  for (auto func : op->getRegion(0).getOps<func::FuncOp>()) {
    if (func.isPrivate())
      continue;

    FailureOr<Offset<mtrt::flat::FunctionSignature>> offt;
    if (auto metaAttr = func->getAttrOfType<executor::FunctionMetadataAttr>(
            executor::ExecutorDialect::kFunctionMetadataAttrName)) {
      offt = translateSignature(fbBuilder, metaAttr);
    } else {
      offt = generateSignature(fbBuilder, func.getFunctionType());
    }
    if (failed(offt))
      return failure();

    Offset<fb::String> funcNameOffset =
        fbBuilder.CreateString(func.getName().str());

    funcOffsets.push_back(
        mtrt::flat::CreateFunction(fbBuilder, funcNameOffset, *offt));
  }

  // Get the process grid by default we use a 2D process grid of shape (1, 1) if
  // the current one is empty. This aligns with what is expected from compiling
  // a StableHLO program (since the StableHLO spec requires a 2D process grid
  // [num_replicas, num_partitions]).
  FailureOr<SmallVector<int64_t>> processGrid =
      executor::getModuleProcessGridShape(op);
  if (failed(processGrid) || processGrid->empty())
    processGrid = SmallVector<int64_t>(2, 1);

  SmallVector<uint32_t> gridShapeU = llvm::map_to_vector(
      *processGrid, [](int64_t x) -> uint32_t { return x; });

  auto constVecOffsets = fbBuilder.serialize(constantOffsets);
  auto vecFuncOffsets = fbBuilder.serialize(funcOffsets);
  auto processGridShapeOffset = fbBuilder.serialize(gridShapeU);
  llvm::StringRef moduleName = op->hasAttr(SymbolTable::getSymbolAttrName())
                                   ? SymbolTable::getSymbolName(op).strref()
                                   : "unnamed-module";
  auto nameOffset = fbBuilder.CreateString(moduleName.str());
  mtrt::flat::ExecutableBuilder exeBuilder(fbBuilder);
  exeBuilder.add_process_grid_shape(processGridShapeOffset);
  exeBuilder.add_functions(vecFuncOffsets);
  exeBuilder.add_data_segments(constVecOffsets);
  exeBuilder.add_source(sourceStrOffset);
  exeBuilder.add_name(nameOffset);
  fbBuilder.Finish(exeBuilder.Finish());

  flatbuffers::DetachedBuffer detached = fbBuilder.Release();

  // Validate the created buffer.
  {
    flatbuffers::Verifier::Options verifierOptions{};
    verifierOptions.max_size = FLATBUFFERS_MAX_64_BUFFER_SIZE;
    flatbuffers::Verifier verifier(detached.data(), detached.size(),
                                   verifierOptions);
    if (!mtrt::flat::VerifyExecutableBuffer(verifier))
      return emitError(op->getLoc())
             << "failed to create a valid Executable buffer";
  }

  std::unique_ptr<mtrt::ExecutableStorage> result =
      std::make_unique<ExecutableStorageFlatbuffer>(std::move(detached));
  return result;
}

LogicalResult mlir::translateToRuntimeExecutable(Operation *op,
                                                 raw_ostream &os) {
  FailureOr<std::unique_ptr<mtrt::ExecutableStorage>> storage =
      translateToRuntimeExecutable(op);
  if (failed(storage) || !*storage)
    return failure();

  os.write(reinterpret_cast<const char *>((*storage)->data()),
           (*storage)->size());
  return success();
}

void mlir::registerToRuntimeExecutableTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-runtime-executable",
      "translate from MLIR to Executor runtime executable",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToRuntimeExecutable(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, cf::ControlFlowDialect,
                        executor::ExecutorDialect, DLTIDialect>();
      });
}
