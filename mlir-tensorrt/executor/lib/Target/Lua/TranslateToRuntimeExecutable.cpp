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
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Target/Lua/TranslateToLua.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace rt = mlirtrt::runtime;

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

class FBBuilder : public fb::FlatBufferBuilder64 {
public:
  template <typename T>
  auto serialize(const std::vector<T> &span) {
    return this->CreateVector(span);
  }

  template <typename T>
  auto serialize(mlir::ArrayRef<T> span) {
    return this->CreateVector(span.data(), span.size());
  }

  template <typename T>
  auto serialize(mlir::SmallVector<T> span) {
    return this->serialize(ArrayRef(span));
  }

  template <typename T>
  auto serialize64(mlir::ArrayRef<T> span) {
    return this->CreateVector64(span.data(), span.size());
  }
};

template <>
auto FBBuilder::serialize64(mlir::ArrayRef<char> span) {
  return this->CreateVector<int8_t, fb::Offset64, fb::Vector64>(
      reinterpret_cast<const int8_t *>(span.data()), span.size());
}

/// An implementation of `ExecutableStorage` that just uses a
/// `flatbuffers::DetachedBuffer`. This allows us to avoid a copy of the
/// serialized buffer.
class ExecutableStorageFlatbuffer : public mlirtrt::runtime::ExecutableStorage {
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

/// Translate the scalar type into the equivalent flatbuffer API object.
static FailureOr<rt::ScalarTypeCode> translateScalarType(Type t) {
  if (t.isInteger(32))
    return rt::ScalarTypeCode::i32;
  if (t.isInteger(16))
    return rt::ScalarTypeCode::i16;
  if (t.isInteger(64))
    return rt::ScalarTypeCode::i64;
  if (t.isInteger(8))
    return rt::ScalarTypeCode::i8;
  if (t.isF32())
    return rt::ScalarTypeCode::f32;
  if (t.isF64())
    return rt::ScalarTypeCode::f64;
  if (t.isF16())
    return rt::ScalarTypeCode::f16;
  if (t.isBF16())
    return rt::ScalarTypeCode::bf16;
  if (t.isFloat8E4M3FN())
    return rt::ScalarTypeCode::f8e4m3fn;
  if (t.isIndex())
    return rt::ScalarTypeCode::i32;
  if (t.isInteger(1))
    return rt::ScalarTypeCode::i1;
  if (t.isInteger(4))
    return rt::ScalarTypeCode::i4;
  return failure();
}

/// Serialize `elAttr` to `output` if `elAttr` is not a splat-type attribute.
static FailureOr<Offset64<fb::Vector64<int8_t>>>
serializeDenseElementsAttr(FBBuilder &fbBuilder,
                           DenseIntOrFPElementsAttr elAttr) {
  assert(!elAttr.isSplat() && "expected non-splat attr");
  unsigned elementBitWidth = elAttr.getElementType().getIntOrFloatBitWidth();
  if (elementBitWidth % 8 == 0)
    return fbBuilder.serialize64(elAttr.getRawData());

  if (elAttr.getElementType().isInteger(1)) {
    auto range = llvm::map_range(elAttr.getValues<bool>(), [](bool inp) {
      return static_cast<int8_t>(inp);
    });
    return fbBuilder.CreateVector64(
        std::vector<int8_t>(range.begin(), range.end()));
  }
  if (elAttr.getElementType().isInteger(4)) {
    auto range = llvm::map_range(elAttr.getValues<APInt>(), [](APInt inp) {
      return static_cast<int8_t>(inp.getSExtValue());
    });
    return fbBuilder.CreateVector64(
        std::vector<int8_t>(range.begin(), range.end()));
  }
  return failure();
}

/// Serialize `elAttr` to `output` if `elAttr` is a splat-type attribute.
static FailureOr<Offset64<fb::Vector64<int8_t>>>
serializeDenseSplatElementsAttr(FBBuilder &fbBuilder,
                                DenseIntOrFPElementsAttr elAttr) {
  assert(elAttr.isSplat() && "expected non-splat attr");
  unsigned elementBitWidth = elAttr.getElementType().getIntOrFloatBitWidth();
  if (elementBitWidth % 8 == 0) {
    auto data = elAttr.getRawData();
    unsigned numBytes = data.size() * elAttr.getNumElements();
    std::vector<int8_t> output;
    output.reserve(numBytes);
    for (int64_t i = 0; i < elAttr.getNumElements(); i++)
      llvm::append_range(output, data);
    return fbBuilder.CreateVector64(output);
  }
  if (elAttr.getElementType().isInteger(1))
    return fbBuilder.CreateVector64(std::vector<int8_t>(
        elAttr.getNumElements(), elAttr.getSplatValue<bool>() ? 1 : 0));
  if (elAttr.getElementType().isInteger(4)) {
    APInt elAttrValue = elAttr.getSplatValue<APInt>();
    return fbBuilder.CreateVector64(
        std::vector<int8_t>(elAttr.getNumElements(),
                            static_cast<int8_t>(elAttrValue.getSExtValue())));
  }
  return failure();
}

/// Return the serialized bytes of the given `attr` and `symbolName`. Note that
/// this only handles bitwidths that are a multiple of 8 (other bit widths need
/// a load/store convention), for e.g. boolean constants or i4 types, etc. It
/// also assumes the endianness matches the host.
/// TODO: Can we replace this with something more robust from upstream?
static FailureOr<Offset64Pair<fb::String, fb::Vector64<int8_t>>>
serializeElementsAttr(FBBuilder &fbBuilder, StringRef symbolName,
                      ElementsAttr attr) {
  // auto serializedConstant = std::make_unique<rt::SerializedConstant>();
  // serializedConstant->name = symbolName;
  auto name = fbBuilder.CreateString<Offset64>(symbolName.str());
  auto retError = [&](StringRef msg) {
    return emitError(UnknownLoc::get(attr.getContext())) << msg;
  };

  TypedAttr typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr)
    return retError("can only serialized typed attributes");

  // Calculate serialization size and allocate buffer.
  Type type = typedAttr.getType();
  unsigned expectedSerializationSize = 0;
  constexpr unsigned kBitsPerByte = 8;
  if (ShapedType shapedType = dyn_cast<ShapedType>(type)) {
    assert(shapedType.hasStaticShape() && "expected static shape");
    expectedSerializationSize = llvm::divideCeil(
        shapedType.getNumElements() * shapedType.getElementTypeBitWidth(),
        kBitsPerByte);
  } else if (type.isIntOrFloat()) {
    expectedSerializationSize =
        llvm::divideCeil(type.getIntOrFloatBitWidth(), kBitsPerByte);
  } else {
    return retError("unhandled attribute type for serialization");
  }

  // Encode resource elements attrs.
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    if (resourceAttr.getElementType().getIntOrFloatBitWidth() % 8 != 0)
      return retError("unhandled resource serialization case");
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    ArrayRef<char> data = handle.getResource()->getBlob()->getData();
    if (data.size() != expectedSerializationSize)
      return retError("unexpected serialization size");
    return std::make_pair(name, fbBuilder.serialize64(data));
  }

  // Encode dense elements attrs.
  if (auto elAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr)) {
    if (!elAttr.isSplat()) {
      auto result = serializeDenseElementsAttr(fbBuilder, elAttr);
      if (failed(result))
        return failure();
      return std::make_pair(name, *result);
    }
    auto result = serializeDenseSplatElementsAttr(fbBuilder, elAttr);
    if (failed(result))
      return failure();
    return std::make_pair(name, *result);
  }
  return retError("unhandled serialization case");
}

/// Serialize the given attribute into the flatbuffer as a Union object. This
/// returns two offsets (one for Bounds enum code, another for the actual
/// concrete bounds object)
static FailureOr<UnionOffset<rt::impl::Bounds>>
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
        rt::impl::Bounds::DimensionBounds,
        rt::impl::CreateDimensionBounds(fb, min, max).Union());
  }

  if (auto vals = llvm::dyn_cast<executor::ValueBoundsAttr>(attr)) {
    auto elemT = vals.getMin().getElementType();
    assert(elemT == vals.getMax().getElementType());
    if (!elemT.isInteger(64))
      return emitError(UnknownLoc::get(attr.getContext()))
             << "Unsupported element type " << elemT << " for attribute ("
             << attr << ") in function metadata";

    auto min = fb.serialize<int64_t>(
        llvm::to_vector(vals.getMin().getValues<int64_t>()));
    auto max = fb.serialize<int64_t>(
        llvm::to_vector(vals.getMax().getValues<int64_t>()));
    return std::make_pair(rt::impl::Bounds::ValueBounds,
                          rt::impl::CreateValueBounds(fb, min, max).Union());
  }

  assert(isa<UnitAttr>(attr) && "Must be a unit attribute");
  return std::make_pair(rt::impl::Bounds::NoneBounds,
                        rt::impl::CreateNoneBounds(fb).Union());
}

/// Translate the memory type into the equivalent flatbuffer API object.
static FailureOr<rt::PointerType> translateMemoryType(executor::MemoryType t) {
  switch (t) {
  case executor::MemoryType::host:
    return rt::PointerType::host;
  case executor::MemoryType::host_pinned:
    return rt::PointerType::pinned_host;
  case executor::MemoryType::device:
    return rt::PointerType::device;
  case executor::MemoryType::unified:
    return rt::PointerType::unified;
  default:
    return failure();
  }
}

/// Serialize the given type into the flatbuffer as a Union object. This returns
/// two offsets (one for Type enum code, another for the actual concrete type
/// object.)
static FailureOr<UnionOffset<rt::impl::Type>>
translateTypeVariant(FBBuilder &fbBuilder, Type t) {
  auto emitTranslateFailure = [&](Type t) {
    return emitError(UnknownLoc::get(t.getContext()))
           << "unhandled type (" << t << ") in Executor function metadata";
  };

  if (!isa<MemRefType, IntegerType, FloatType, executor::ExecutorOpaqueType>(t))
    return emitTranslateFailure(t);

  // Encode as a memref.
  if (auto memrefType = llvm::dyn_cast<MemRefType>(t)) {
    FailureOr<rt::ScalarTypeCode> code =
        translateScalarType(memrefType.getElementType());
    if (failed(code))
      return emitTranslateFailure(memrefType.getElementType());
    auto shape = fbBuilder.serialize<int64_t>(memrefType.getShape());
    auto [strides, offset] = mlir::getStridesAndOffset(memrefType);
    auto stridesOffset = fbBuilder.serialize<int64_t>(strides);

    auto addressSpace = rt::PointerType::unknown;
    if (memrefType.getMemorySpace()
            .isa_and_nonnull<executor::MemoryTypeAttr>()) {
      auto memoryType =
          executor::PointerType::get(memrefType.getContext(),
                                     memrefType.getMemorySpace()
                                         .dyn_cast<executor::MemoryTypeAttr>()
                                         .getValue())
              .getAddressSpace();
      addressSpace = *translateMemoryType(memoryType);
    }
    return std::make_pair(rt::impl::Type::MemRefType,
                          rt::impl::CreateMemRefType(fbBuilder, *code, shape,
                                                     stridesOffset,
                                                     addressSpace)
                              .Union());
  }

  // Encode as opaque external reference.
  if (auto opaqueType = llvm::dyn_cast<executor::ExecutorOpaqueType>(t)) {

    // TODO: replace hard-coded name with better logic. This should be part of
    // the runner/executable API.
    rt::impl::ExternalOpaqueRefKind refKind =
        rt::impl::ExternalOpaqueRefKind::unknown;
    if (opaqueType.getName() == "cuda_stream")
      refKind = rt::impl::ExternalOpaqueRefKind::cuda_stream;

    return std::make_pair(
        rt::impl::Type::ExternalOpaqueRefType,
        rt::impl::CreateExternalOpaqueRefType(fbBuilder, refKind).Union());
  }

  // Encode as a scalar type.
  FailureOr<rt::ScalarTypeCode> code = translateScalarType(t);
  if (failed(code))
    return emitTranslateFailure(t);
  return std::make_pair(rt::impl::Type::ScalarType,
                        rt::impl::CreateScalarType(fbBuilder, *code).Union());
}

/// Translate the calling convention.
static rt::impl::CallingConvention
translateCallingConvention(executor::CallingConvention cconv) {
  switch (cconv) {
  case executor::CallingConvention::packed:
    return rt::impl::CallingConvention::packed;
  case executor::CallingConvention::unpacked:
    return rt::impl::CallingConvention::unpacked;
  }
  llvm_unreachable(
      "unknown MLIR Executor -> MTRT runtime calling convention translation");
}

/// Encode the FunctionSignature into the flatbuffer and return the offset of
/// the serialized data.
static FailureOr<Offset<rt::impl::FunctionSignature>>
translateSignature(FBBuilder &fbBuilder,
                   executor::FunctionMetadataAttr metadata) {
  // Union type must be encoded as variant type + data.
  SmallVector<rt::impl::Type> argVariantTypes, resultVariantTypes;
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

  SmallVector<rt::impl::Bounds> argBounds, resBounds;
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

  return rt::impl::CreateFunctionSignature(
      fbBuilder, fbBuilder.serialize(argVariantTypes),
      fbBuilder.serialize(argOffsets), fbBuilder.serialize(resultVariantTypes),
      fbBuilder.serialize(resultOffsets), metadata.getNumOutputArgs(), fbBounds,
      fbBoundsOffsets, fbBuilder.serialize(resBounds),
      fbBuilder.serialize(resBoundsOffsets), shapeFuncSym,
      translateCallingConvention(metadata.getCconv()));
}

FailureOr<std::unique_ptr<mlirtrt::runtime::ExecutableStorage>>
mlir::translateToRuntimeExecutable(Operation *op) {

  FBBuilder fbBuilder;
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return failure();

  //===----------------------------------------------------------------------===//
  // 64 bit section
  //===----------------------------------------------------------------------===//

  // For each `executor.constant_resource` operation, if there is a constant
  // data value attached to it, then serialize that constant data in the
  // executable as a Constant. These go into the 64bit section. We serialize the
  // string with the data in the 64 bit section.
  SmallVector<Offset64Pair<fb::String, fb::Vector64<int8_t>>> constData;
  for (auto resourceOp : module.getOps<executor::ConstantResourceOp>()) {

    auto serializedAttr = serializeElementsAttr(fbBuilder, resourceOp.getName(),
                                                resourceOp.getValue());
    if (failed(serializedAttr))
      return resourceOp->emitOpError("failed to encode constant value " +
                                     Twine(resourceOp.getSymName()) +
                                     " as a SerializedConstant");
    constData.push_back(*serializedAttr);
  }

  //===----------------------------------------------------------------------===//
  // 32 bit section
  //===----------------------------------------------------------------------===//
  SmallVector<Offset<rt::impl::Constant>> constantOffsets;
  constantOffsets.reserve(constData.size());
  for (const auto &[strOffset, dataOffset] : constData)
    constantOffsets.push_back(
        rt::impl::CreateConstant(fbBuilder, strOffset, dataOffset));

  std::string sourceString;
  {
    llvm::raw_string_ostream ss(sourceString);
    if (failed(mlir::translateToLua(op, ss)))
      return emitError(op->getLoc(), "Lua translation failed");
  }
  Offset<fb::String> sourceStrOffset = fbBuilder.CreateString(sourceString);

  // Loop over all functions and collect metadata (function names and
  // signatures) that we will embed in the executable.
  SmallVector<Offset<rt::impl::Function>> funcOffsets;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isPrivate())
      continue;
    auto metaAttr = func->getAttrOfType<executor::FunctionMetadataAttr>(
        executor::ExecutorDialect::kFunctionMetadataAttrName);
    bool isTrivialFunc =
        func.getNumArguments() == 0 || func.getNumResults() == 0;
    if (!metaAttr && !isTrivialFunc)
      return func->emitError(
          "non-trivial functions must have a executor.function_metadata "
          "attribute for translation");

    Offset<fb::String> funcNameOffset =
        fbBuilder.CreateString(func.getName().str());

    // Encode the signature metadata if present.
    Offset<rt::impl::FunctionSignature> sigOffset = 0;
    if (metaAttr) {
      FailureOr<Offset<rt::impl::FunctionSignature>> offt =
          translateSignature(fbBuilder, metaAttr);
      if (failed(offt))
        return failure();
      sigOffset = std::move(*offt);
    } else {
      sigOffset = rt::impl::CreateFunctionSignature(fbBuilder);
    }
    funcOffsets.push_back(
        rt::impl::CreateFunction(fbBuilder, funcNameOffset, sigOffset));
  }

  // Get the process grid by default we use a 2D process grid of shape (1, 1) if
  // the current one is empty. This aligns with what is expected from compiling
  // a StableHLO program (since the StableHLO spec requires a 2D process grid
  // [num_replicas, num_partitions]).
  FailureOr<SmallVector<int64_t>> processGrid =
      executor::getModuleProcessGridShape(module);
  if (failed(processGrid) || processGrid->empty())
    processGrid = SmallVector<int64_t>(2, 1);

  SmallVector<uint32_t> gridShapeU = llvm::map_to_vector(
      *processGrid, [](int64_t x) -> uint32_t { return x; });

  auto constVecOffsets = fbBuilder.serialize(constantOffsets);
  auto vecFuncOffsets = fbBuilder.serialize(funcOffsets);
  auto processGridShapeOffset = fbBuilder.serialize(gridShapeU);
  llvm::StringRef moduleName =
      module.getSymName() ? *module.getSymName() : "unnamed-module";
  auto nameOffset = fbBuilder.CreateString(moduleName.str());
  rt::impl::ExecutableBuilder exeBuilder(fbBuilder);
  exeBuilder.add_process_grid_shape(processGridShapeOffset);
  exeBuilder.add_functions(vecFuncOffsets);
  exeBuilder.add_constants(constVecOffsets);
  exeBuilder.add_source(sourceStrOffset);
  exeBuilder.add_name(nameOffset);
  fbBuilder.Finish(exeBuilder.Finish());

  flatbuffers::DetachedBuffer detached = fbBuilder.Release();

  std::unique_ptr<mlirtrt::runtime::ExecutableStorage> result =
      std::make_unique<ExecutableStorageFlatbuffer>(std::move(detached));
  return result;
}

LogicalResult mlir::translateToRuntimeExecutable(Operation *op,
                                                 raw_ostream &os) {
  FailureOr<std::unique_ptr<mlirtrt::runtime::ExecutableStorage>> storage =
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
