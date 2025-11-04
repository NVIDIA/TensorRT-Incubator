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
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
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
static FailureOr<mtrt::ScalarTypeCode>
translateScalarType(Type t, const mlir::DataLayout &dataLayout) {
  if (isa<IndexType>(t)) {
    uint64_t indexBitwidth = dataLayout.getTypeSizeInBits(t);
    if (indexBitwidth == 32)
      return mtrt::ScalarTypeCode::i32;
    if (indexBitwidth == 64)
      return mtrt::ScalarTypeCode::i64;
    return failure();
  }

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

/// Helper to createninj a BoundsUnion from DimensionBoundsT.
static mtrt::flat::BoundsUnion
createBoundsUnion(mtrt::flat::DimensionBoundsT &&bounds) {
  mtrt::flat::BoundsUnion boundsUnion;
  boundsUnion.Set(std::move(bounds));
  return boundsUnion;
}

/// Helper to create a BoundsUnion from ValueBoundsT.
static mtrt::flat::BoundsUnion
createBoundsUnion(mtrt::flat::ValueBoundsT &&bounds) {
  mtrt::flat::BoundsUnion boundsUnion;
  boundsUnion.Set(std::move(bounds));
  return boundsUnion;
}

/// Translate the given attribute into a native Object API BoundsUnion.
/// Returns a BoundsUnion with type NONE for UnitAttr.
static FailureOr<mtrt::flat::BoundsUnion>
translateBoundsAttribute(Attribute attr) {
  if (!isa<executor::DimensionBoundsAttr, executor::ValueBoundsAttr, UnitAttr>(
          attr))
    return emitError(UnknownLoc::get(attr.getContext()))
           << "unhandled attribute (" << attr
           << ") in Executor function metadata";

  if (auto dims = llvm::dyn_cast<executor::DimensionBoundsAttr>(attr)) {
    mtrt::flat::DimensionBoundsT dimensionBounds;
    dimensionBounds.min = {dims.getMin().asArrayRef().begin(),
                           dims.getMin().asArrayRef().end()};
    dimensionBounds.max = {dims.getMax().asArrayRef().begin(),
                           dims.getMax().asArrayRef().end()};
    return createBoundsUnion(std::move(dimensionBounds));
  }

  if (auto vals = llvm::dyn_cast<executor::ValueBoundsAttr>(attr)) {
    auto toI64 = [](const llvm::APInt &v) { return v.getSExtValue(); };
    mtrt::flat::ValueBoundsT valueBounds;
    auto minVec = llvm::map_to_vector(vals.getMin().getValues<APInt>(), toI64);
    auto maxVec = llvm::map_to_vector(vals.getMax().getValues<APInt>(), toI64);
    valueBounds.min = {minVec.begin(), minVec.end()};
    valueBounds.max = {maxVec.begin(), maxVec.end()};
    return createBoundsUnion(std::move(valueBounds));
  }

  assert(isa<UnitAttr>(attr) && "Must be a unit attribute");
  // Return a BoundsUnion with NONE type for UnitAttr
  return mtrt::flat::BoundsUnion();
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

/// Helper to create a TypeUnion from ScalarTypeT.
static mtrt::flat::TypeUnion
createTypeUnion(mtrt::flat::ScalarTypeT &&scalarType) {
  mtrt::flat::TypeUnion typeUnion;
  typeUnion.Set(std::move(scalarType));
  return typeUnion;
}

/// Helper to create a TypeUnion from MemRefTypeT.
static mtrt::flat::TypeUnion
createTypeUnion(mtrt::flat::MemRefTypeT &&memrefType) {
  mtrt::flat::TypeUnion typeUnion;
  typeUnion.Set(std::move(memrefType));
  return typeUnion;
}

/// Translate the given type into a native Object API TypeUnion.
static FailureOr<mtrt::flat::TypeUnion>
translateTypeVariant(Type t, const mlir::DataLayout &dataLayout) {
  auto emitTranslateFailure = [&](Type t) {
    return emitError(UnknownLoc::get(t.getContext()))
           << "unhandled type (" << t << ") in Executor function metadata";
  };

  if (!isa<MemRefType, IntegerType, FloatType, Float8E4M3FNType,
           Float4E2M1FNType, ComplexType, IndexType>(t))
    return emitTranslateFailure(t);

  // Encode as a memref.
  if (auto memrefType = llvm::dyn_cast<MemRefType>(t)) {
    FailureOr<mtrt::ScalarTypeCode> code =
        translateScalarType(memrefType.getElementType(), dataLayout);
    if (failed(code))
      return emitTranslateFailure(memrefType.getElementType());

    auto [strides, offset] = memrefType.getStridesAndOffset();

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

    mtrt::flat::MemRefTypeT memref;
    memref.element_type = *code;
    memref.shape = {memrefType.getShape().begin(), memrefType.getShape().end()};
    memref.strides = {strides.begin(), strides.end()};
    memref.address_space = addressSpace;
    return createTypeUnion(std::move(memref));
  }

  // Encode as a scalar type.
  FailureOr<mtrt::ScalarTypeCode> code = translateScalarType(t, dataLayout);
  if (failed(code))
    return emitTranslateFailure(t);

  mtrt::flat::ScalarTypeT scalar;
  scalar.type = *code;
  return createTypeUnion(std::move(scalar));
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

/// Build a FunctionSignature using the Object API.
static FailureOr<mtrt::flat::FunctionSignatureT>
translateSignature(executor::FunctionMetadataAttr metadata,
                   const mlir::DataLayout &dataLayout) {
  assert(metadata && "expected valid FunctionMetadataAttr");

  mtrt::flat::FunctionSignatureT signature;

  // Translate argument types
  for (Type t : metadata.getArgs()) {
    auto typeUnion = translateTypeVariant(t, dataLayout);
    if (failed(typeUnion))
      return failure();
    signature.args.push_back(std::move(*typeUnion));
  }

  // Translate result types
  for (Type t : metadata.getResults()) {
    auto typeUnion = translateTypeVariant(t, dataLayout);
    if (failed(typeUnion))
      return failure();
    signature.results.push_back(std::move(*typeUnion));
  }

  // Translate bounds - collect non-NONE bounds and create index mappings
  std::vector<mtrt::flat::BoundsUnion> boundsValues;

  // Process argument bounds
  for (Attribute a : metadata.getArgBounds()) {
    auto boundsUnion = translateBoundsAttribute(a);
    if (failed(boundsUnion))
      return failure();

    if (boundsUnion->type != mtrt::flat::Bounds::NONE) {
      signature.arg_bounds_indices.push_back(boundsValues.size());
      boundsValues.push_back(std::move(*boundsUnion));
    } else {
      signature.arg_bounds_indices.push_back(-1);
    }
  }

  // Process result bounds
  for (Attribute a : metadata.getResultBounds()) {
    auto boundsUnion = translateBoundsAttribute(a);
    if (failed(boundsUnion))
      return failure();

    if (boundsUnion->type != mtrt::flat::Bounds::NONE) {
      signature.result_bounds_indices.push_back(boundsValues.size());
      boundsValues.push_back(std::move(*boundsUnion));
    } else {
      signature.result_bounds_indices.push_back(-1);
    }
  }

  signature.bounds_values = std::move(boundsValues);
  signature.num_output_args = metadata.getNumOutputArgs();
  signature.shape_function_name =
      metadata.getShapeFunc() ? metadata.getShapeFunc().getAttr().str() : "";
  signature.calling_convention =
      translateCallingConvention(metadata.getCconv());

  return signature;
}

/// Generate an empty function signature. This is used if there is no explicit
/// 'executor.function_metadata' attached to the function.
static mtrt::flat::FunctionSignatureT generateSignature() {
  mtrt::flat::FunctionSignatureT signature;
  signature.num_output_args = 0;
  signature.calling_convention = mtrt::CallingConvention::unpacked;
  return signature;
}

LogicalResult
translateBoundsIfPresent(func::FuncOp func, unsigned argIndex,
                         mtrt::flat::FunctionSignatureT &signature,
                         bool isInput) {
  for (llvm::StringRef attrName :
       {executor::ExecutorDialect::getShapeBoundsAttrName(),
        executor::ExecutorDialect::getValueBoundsAttrName()}) {
    if (Attribute bounds = func.getArgAttr(argIndex, attrName)) {
      auto boundsUnion = translateBoundsAttribute(bounds);
      if (failed(boundsUnion))
        return failure();
      unsigned boundsIdx = signature.bounds_values.size();
      if (isInput) {
        signature.arg_bounds_indices.push_back(boundsIdx);
      } else {
        signature.result_bounds_indices.push_back(boundsIdx);
      }
      signature.bounds_values.push_back(std::move(*boundsUnion));
      return success();
    }
  }
  if (isInput)
    signature.arg_bounds_indices.push_back(-1);
  else
    signature.result_bounds_indices.push_back(-1);
  return success();
}

/// Build a FunctionSignature for an ABI wrapper function by extracting
/// information from the ArgumentABIAttr attributes on function arguments.
static FailureOr<mtrt::flat::FunctionSignatureT>
translateABIWrapperSignature(func::FuncOp func,
                             const mlir::DataLayout &dataLayout) {
  mtrt::flat::FunctionSignatureT signature;

  // ABI wrapper functions use unpacked calling convention
  signature.calling_convention = mtrt::CallingConvention::unpacked;
  signature.shape_function_name = "";
  if (auto shapeFunc = func->getAttrOfType<SymbolRefAttr>(
          executor::ExecutorDialect::kShapeFuncAttrName))
    signature.shape_function_name = shapeFunc.getLeafReference().str();

  auto hostPointerType =
      executor::PointerType::get(func.getContext(), executor::MemoryType::host);

  // Process each argument
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    BlockArgument arg = func.getArgument(i);

    std::optional<unsigned> resultIdx =
        executor::abi::isOutputArgument(func, arg);
    const bool isInput = !resultIdx.has_value();
    if (executor::abi::isScalarArgumentType(arg.getType())) {
      assert(isInput && "scalar arguments cannot be output arguments");
      auto typeUnion = translateTypeVariant(arg.getType(), dataLayout);
      if (failed(typeUnion))
        return failure();
      signature.args.push_back(std::move(*typeUnion));
      signature.arg_bounds_indices.push_back(-1);
      continue;
    }

    if (arg.getType() != hostPointerType)
      return emitError(func.getLoc())
             << "ABI wrapper function argument " << i
             << " has an invalid type: " << arg.getType();

    // Get the ArgumentABIAttr for this argument
    executor::ArgumentABIAttr abiAttr =
        executor::abi::getArgumentABIAttr(func, arg);
    if (!abiAttr)
      return emitError(func.getLoc()) << "ABI wrapper function missing "
                                         "executor.abi attribute on argument "
                                      << i;

    if (failed(translateBoundsIfPresent(func, i, signature, isInput)))
      return failure();

    // Extract the value type from the ABI attribute
    Type valueType = abiAttr.getValueType();
    auto typeUnion = translateTypeVariant(valueType, dataLayout);
    if (failed(typeUnion))
      return failure();

    if (isInput) {
      signature.args.push_back(std::move(*typeUnion));
    } else {
      signature.results.push_back(std::move(*typeUnion));
      signature.undef.push_back(abiAttr.getUndef());
    }
  }

  signature.num_output_args = signature.results.size();
  return signature;
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
      if (auto elementsAttr =
              llvm::dyn_cast<ElementsAttr>(resourceOp.getValueAttr())) {
        FailureOr<Offset64<fb::Vector64<int8_t>>> serializedAttr =
            fbBuilder.serialize64<int8_t>(resourceOp.getLoc(), dataLayout,
                                          elementsAttr);
        if (failed(serializedAttr))
          return resourceOp->emitOpError("failed to encode constant value " +
                                         Twine(resourceOp.getSymName()) +
                                         " as a SerializedConstant");
        constData.push_back(*serializedAttr);
      } else if (auto stringAttr =
                     llvm::dyn_cast<StringAttr>(resourceOp.getValueAttr())) {
        llvm::StringRef strref = stringAttr.strref();
        FailureOr<Offset64<fb::Vector64<int8_t>>> serializedAttr =
            fbBuilder.serialize64<int8_t>(llvm::ArrayRef<int8_t>(
                reinterpret_cast<const int8_t *>(strref.data()),
                strref.size()));
        if (failed(serializedAttr))
          return resourceOp->emitOpError("failed to encode constant value " +
                                         Twine(resourceOp.getSymName()) +
                                         " as a SerializedConstant");
        constData.push_back(*serializedAttr);
      } else {
        llvm_unreachable("expected elements or string attribute");
      }
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
    uint64_t align = dataLayout.getTypeABIAlignment(globalOp.getElementType());
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

  // First pass: Check that we don't have a mix of ABI wrapper and non-ABI
  // wrapper functions. Validate that all public functions are either all ABI
  // wrapper functions or all non-ABI wrapper functions.
  bool hasABIWrapperFunctions = false;
  bool hasNonABIWrapperFunctions = false;
  for (auto func : op->getRegion(0).getOps<func::FuncOp>()) {
    if (func.isPrivate())
      continue;

    if (executor::abi::isABIWrapperFunction(func))
      hasABIWrapperFunctions = true;
    else
      hasNonABIWrapperFunctions = true;

    if (hasABIWrapperFunctions && hasNonABIWrapperFunctions)
      return emitError(op->getLoc())
             << "module contains a mix of ABI wrapper functions and non-ABI "
                "wrapper functions; all public functions must be either ABI "
                "wrapper functions or non-ABI wrapper functions";
  }

  // Loop over all functions and collect metadata (function names and
  // signatures) that we will embed in the executable.
  SmallVector<Offset<mtrt::flat::Function>> funcOffsets;
  uint32_t abiVersion = hasABIWrapperFunctions ? 1 : 0;
  for (auto func : op->getRegion(0).getOps<func::FuncOp>()) {
    if (func.isPrivate())
      continue;

    // Build Function using Object API
    mtrt::flat::FunctionT function;
    function.name = func.getName().str();
    function.abi_version = abiVersion;

    // Check if this is an ABI wrapper function
    if (executor::abi::isABIWrapperFunction(func)) {
      // For ABI wrapper functions, extract signature from argument attributes
      auto sig = translateABIWrapperSignature(func, dataLayout);
      if (failed(sig))
        return failure();
      sig->abi_version = abiVersion;
      function.signature =
          std::make_unique<mtrt::flat::FunctionSignatureT>(std::move(*sig));
    } else if (auto metaAttr =
                   func->getAttrOfType<executor::FunctionMetadataAttr>(
                       executor::ExecutorDialect::kFunctionMetadataAttrName)) {
      auto sig = translateSignature(metaAttr, dataLayout);
      if (failed(sig))
        return failure();
      sig->abi_version = abiVersion;
      function.signature =
          std::make_unique<mtrt::flat::FunctionSignatureT>(std::move(*sig));
    } else {
      auto sig = generateSignature();
      sig.abi_version = abiVersion;
      function.signature =
          std::make_unique<mtrt::flat::FunctionSignatureT>(std::move(sig));
    }

    // Pack the Function object into the flatbuffer
    funcOffsets.push_back(mtrt::flat::CreateFunction(fbBuilder, &function));
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
  // Set ABI version to 1 if we have any ABI wrapper functions, otherwise 0
  exeBuilder.add_abi_version(hasABIWrapperFunctions ? 1 : 0);
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
