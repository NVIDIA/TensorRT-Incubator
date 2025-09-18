//===- Common.cpp -----0---------------------------------------------------===//
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
///
///
//===----------------------------------------------------------------------===//
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-tensorrt-common-c/Support/Status.h"

using namespace mtrt;
using namespace mtrt;

#define DEFINE_C_API_PTR_METHODS(name, cpptype)                                \
  static inline name wrap(cpptype *cpp) { return name{cpp}; }                  \
  static inline cpptype *unwrap(name c) {                                      \
    return static_cast<cpptype *>(c.ptr);                                      \
  }

// We just map the CAPI types to Flatbuffer-generated "Object API" types (as
// opposed to the 'view' types that we use in the C++ side).
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
DEFINE_C_API_PTR_METHODS(MTRT_Type, ::mtrt::flat::TypeUnion)
DEFINE_C_API_PTR_METHODS(MTRT_Bounds, ::mtrt::flat::BoundsUnion)
DEFINE_C_API_PTR_METHODS(MTRT_ScalarType, ::mtrt::flat::ScalarTypeT)
DEFINE_C_API_PTR_METHODS(MTRT_MemRefType, ::mtrt::flat::MemRefTypeT)
DEFINE_C_API_PTR_METHODS(MTRT_FunctionSignature,
                         ::mtrt::flat::FunctionSignature)
DEFINE_C_API_PTR_METHODS(MTRT_Executable, ::mtrt::Executable)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

//===----------------------------------------------------------------------===//
// Printing/Formatting Utilities
//===----------------------------------------------------------------------===//

namespace {
/// This is reproduced from MLIR's `mlir::detail::CallbackOstream` from
/// `third_party/llvm-project/mlir/include/mlir/CAPI/Utils.h` in order to break
/// dependence on MLIR headers in this section of the API. A simple raw ostream
/// subclass that forwards write_impl calls to the user-supplied callback
/// together with opaque user-supplied data.
class CallbackOstream : public llvm::raw_ostream {
public:
  CallbackOstream(MTRT_PrintCallbackInfo info)
      : CallbackOstream(info.callback, info.userData) {}
  CallbackOstream(std::function<void(MTRT_StringView, void *)> callback,
                  void *opaqueData)
      : raw_ostream(/*unbuffered=*/true), callback(std::move(callback)),
        opaqueData(opaqueData), pos(0u) {}

  void write_impl(const char *ptr, size_t size) override {
    MTRT_StringView string = mtrtStringViewCreate(ptr, size);
    callback(string, opaqueData);
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

private:
  std::function<void(MTRT_StringView, void *)> callback;
  void *opaqueData;
  uint64_t pos;
};
} // namespace

//===----------------------------------------------------------------------===//
// MTRT_Executable
//===----------------------------------------------------------------------===//

bool mtrtExecutableIsNull(MTRT_Executable executable) {
  return !executable.ptr;
}

MTRT_Status mtrtExecutableDestroy(MTRT_Executable executable) {
  delete reinterpret_cast<Executable *>(executable.ptr);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtExecutableCreate(MTRT_StringView buffer,
                                 MTRT_Executable *result) {
  llvm::ArrayRef<char> data(buffer.data, buffer.length);
  StatusOr<std::unique_ptr<Executable>> executable =
      Executable::loadFromUnalignedRef(data);

  if (!executable.isOk()) {
    auto status = executable.getStatus();
    return mtrtStatusCreate(static_cast<MTRT_StatusCode>(status.getCode()),
                            status.getString().c_str());
  }

  *result = MTRT_Executable{executable->release()};
  return mtrtStatusGetOk();
}

MTRT_Status mtrtExecutableGetStorageView(MTRT_Executable executable,
                                         MTRT_StringView *buffer,
                                         size_t *requiredAlignment) {
  const Executable *exe = unwrap(executable);
  *buffer =
      mtrtStringViewCreate(static_cast<const char *>(exe->getStorage()->data()),
                           exe->getStorage()->size());
  if (requiredAlignment) {
    *requiredAlignment = 8;
  }
  return mtrtStatusGetOk();
}

MTRT_Status mtrtExecutableGetNumDataSegments(MTRT_Executable executable,
                                             int64_t *result) {
  const Executable *exe = unwrap(executable);
  *result = exe->getNumDataSegments();
  return mtrtStatusGetOk();
}

MTRT_Status mtrtExecutableGetDataSegmentInfo(MTRT_Executable executable,
                                             int64_t index,
                                             MTRT_StringView *data,
                                             MTRT_StringView *name) {
  assert(data && name && "expected valid data and name output arguments");
  const Executable *exe = unwrap(executable);
  if (index < 0 || index >= static_cast<int64_t>(exe->getNumDataSegments()))
    return mtrtStatusCreate(MTRT_StatusCode::MTRT_StatusCode_InvalidArgument,
                            "Invalid data segment index");

  const DataSegmentInfo segment = exe->getDataSegments(index);
  *data = mtrtStringViewCreate(reinterpret_cast<const char *>(segment.data()),
                               segment.size());
  const std::string_view segmentName = segment.getName();
  *name = mtrtStringViewCreate(segmentName.data(), segmentName.size());
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_Type
//===----------------------------------------------------------------------===//

bool mtrtTypeIsNull(MTRT_Type type) { return !type.ptr; }

MTRT_Status mtrtTypeDestroy(MTRT_Type type) {
  delete unwrap(type);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_ScalarTypeCode
//===----------------------------------------------------------------------===//

MTRT_Status mtrtScalarTypeCodeBitsPerElement(MTRT_ScalarTypeCode code,
                                             int64_t *result) {
  *result = ScalarType(static_cast<ScalarTypeCode>(code)).getBitWidth();
  return mtrtStatusGetOk();
}

const char *mtrtScalarTypeCodeGetString(MTRT_ScalarTypeCode code) {
  return mtrt::flat::EnumNameScalarTypeCode(static_cast<ScalarTypeCode>(code));
}

//===----------------------------------------------------------------------===//
// MTRT_ScalarType
//===----------------------------------------------------------------------===//

bool mtrtScalarTypeIsNull(MTRT_ScalarType scalar) { return !scalar.ptr; }

MTRT_Status mtrtScalarTypeDestroy(MTRT_ScalarType scalar) {
  delete unwrap(scalar);
  return mtrtStatusGetOk();
}

bool mtrtTypeIsaScalarType(MTRT_Type type) {
  return unwrap(type)->type == mtrt::flat::Type::ScalarType;
}

MTRT_ScalarTypeCode mtrtScalarTypeGetCode(MTRT_ScalarType type) {
  return static_cast<MTRT_ScalarTypeCode>(unwrap(type)->type);
}

MTRT_ScalarType mtrtTypeGetScalarType(MTRT_Type type) {
  assert(mtrtTypeIsaScalarType(type) && "expected ScalarType");
  mtrt::flat::TypeUnion *typeUnion = unwrap(type);
  return wrap(typeUnion->AsScalarType());
}

MTRT_Status mtrtScalarTypeCreate(MTRT_ScalarTypeCode code, MTRT_Type *result) {
  mtrt::flat::ScalarTypeT cppScalarType;
  cppScalarType.type = (static_cast<ScalarTypeCode>(code));

  // Allocate the TypeUnion object, populate it by moving in the
  // concrete object, and release it to be owned by the CAPI object.
  auto cppType = std::make_unique<mtrt::flat::TypeUnion>();
  cppType->Set(std::move(cppScalarType));
  *result = wrap(cppType.release());
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_MemRefType
//===----------------------------------------------------------------------===//

bool mtrtMemRefTypeIsNull(MTRT_MemRefType memref) { return !memref.ptr; }

MTRT_Status mtrtMemRefTypeDestroy(MTRT_MemRefType memref) {
  delete unwrap(memref);
  return mtrtStatusGetOk();
}

bool mtrtTypeIsaMemRefType(MTRT_Type type) {
  return unwrap(type)->type == mtrt::flat::Type::MemRefType;
}

MTRT_Status mtrtMemRefTypeCreate(int64_t rank, const int64_t *shape,
                                 MTRT_ScalarTypeCode elementType,
                                 MTRT_PointerType addressSpace,
                                 MTRT_Type *result) {
  mtrt::flat::MemRefTypeT memref;
  /// TODO: update this so that element type becomes nested Type union.
  memref.element_type = static_cast<ScalarTypeCode>(elementType);
  memref.shape = std::vector(shape, shape + rank);
  memref.address_space = static_cast<mtrt::flat::PointerType>(addressSpace);

  // allocate default strides (can expose a different creation method if needed
  // to set explicitly).
  std::vector<int64_t> strides(rank, 1);
  for (int64_t i = rank - 2; i >= 0; i--)
    strides[i] = strides[i + 1] * shape[i + 1];
  memref.strides = std::move(strides);

  // Allocate the TypeUnion object, populate it by moving in the
  // concrete object, and release it to be owned by the CAPI object.
  auto cppType = std::make_unique<mtrt::flat::TypeUnion>();
  cppType->Set(std::move(memref));
  *result = wrap(cppType.release());
  return mtrtStatusGetOk();
}

MTRT_MemRefType mtrtTypeGetMemRefType(MTRT_Type type) {
  assert(mtrtTypeIsaMemRefType(type) && "expected MemRefType");
  mtrt::flat::TypeUnion *typeUnion = unwrap(type);
  return wrap(typeUnion->AsMemRefType());
}

/// Retrieve metadata for the provided memref.
MTRT_Status mtrtMemRefTypeGetInfo(MTRT_Type memref, MTRT_MemRefTypeInfo *info) {
  mtrt::flat::MemRefTypeT *cppType = unwrap(mtrtTypeGetMemRefType(memref));
  MTRT_MemRefTypeInfo result;
  result.rank = cppType->shape.size();
  result.shape = cppType->shape.data();
  result.strides = cppType->strides.data();
  result.elementType = static_cast<MTRT_ScalarTypeCode>(cppType->element_type);
  result.addressSpace = static_cast<MTRT_PointerType>(cppType->address_space);
  *info = std::move(result);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_Bounds
//===----------------------------------------------------------------------===//

bool mtrtBoundsIsNull(MTRT_Bounds bounds) { return !bounds.ptr; }

MTRT_Status mtrtBoundsDestroy(MTRT_Bounds bounds) {
  delete unwrap(bounds);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtBoundsGetSize(MTRT_Bounds bounds,
                              MTRT_ArrayRefI64 *boundValues) {
  auto b = unwrap(bounds);
  if (auto d = b->AsDimensionBounds()) {
    boundValues->size = d->min.size();
    return mtrtStatusGetOk();
  }
  if (auto d = b->AsValueBounds()) {
    boundValues->size = d->min.size();
    return mtrtStatusGetOk();
  }
  if (b->AsNoneBounds()) {
    boundValues->size = 0;
    return mtrtStatusGetOk();
  }
  assert(b->type == mtrt::flat::Bounds::NONE);
  boundValues->size = 0;
  return mtrtStatusGetOk();
}

MTRT_Status mtrtBoundsGetMin(MTRT_Bounds bounds, MTRT_ArrayRefI64 *minBounds) {
  auto b = unwrap(bounds);
  int64_t size = minBounds->size;
  if (auto d = b->AsDimensionBounds()) {
    for (int i = 0; i < size; ++i)
      minBounds->ptr[i] = d->min[i];
    return mtrtStatusGetOk();
  }
  if (auto d = b->AsValueBounds()) {
    for (int i = 0; i < size; ++i)
      minBounds->ptr[i] = d->min[i];
    return mtrtStatusGetOk();
  }
  if (b->AsNoneBounds()) {
    return mtrtStatusGetOk();
  }
  return mtrtStatusCreate(
      MTRT_StatusCode::MTRT_StatusCode_InvalidArgument,
      "Unable to calculate min bound. Bounds attribute is invalid.");
}

MTRT_Status mtrtBoundsGetMax(MTRT_Bounds bounds, MTRT_ArrayRefI64 *maxBounds) {
  auto b = unwrap(bounds);
  int64_t size = maxBounds->size;
  if (auto d = b->AsDimensionBounds()) {
    for (int i = 0; i < size; ++i)
      maxBounds->ptr[i] = d->max[i];
    return mtrtStatusGetOk();
  }
  if (auto d = b->AsValueBounds()) {
    for (int i = 0; i < size; ++i)
      maxBounds->ptr[i] = d->max[i];
    return mtrtStatusGetOk();
  }
  if (b->AsNoneBounds()) {
    return mtrtStatusGetOk();
  }
  return mtrtStatusCreate(
      MTRT_StatusCode::MTRT_StatusCode_InvalidArgument,
      "Unable to calculate max bound. Bounds attribute is invalid.");
}

//===----------------------------------------------------------------------===//
// MTRT_FunctionSignature
//===----------------------------------------------------------------------===//

MTRT_FunctionSignature mtrtGetFunctionSignature(MTRT_Executable exec,
                                                const char *name) {
  auto sig = const_cast<mtrt::flat::FunctionSignature *>(
      (*unwrap(exec)->getFunction(name)).getSignature().view);
  return wrap(sig);
}

MTRT_Status mtrtFunctionSignatureGetString(MTRT_FunctionSignature signature,
                                           MTRT_PrintCallbackInfo callback) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  CallbackOstream stream(callback);
  mtrt::print(stream, sig);
  return mtrtStatusGetOk();
}

bool mtrtFunctionSignatureIsNull(MTRT_FunctionSignature signature) {
  return !signature.ptr;
}

MTRT_Status mtrtFunctionSignatureGetNumArgs(MTRT_FunctionSignature signature,
                                            int64_t *numArgs) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numArgs = FunctionSignatureView(sig).getNumArgs();
  return mtrtStatusGetOk();
}

MTRT_Status mtrtFunctionSignatureGetNumResults(MTRT_FunctionSignature signature,
                                               int64_t *numResults) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numResults = FunctionSignatureView(sig).getNumResults();
  return mtrtStatusGetOk();
}

MTRT_Status
mtrtFunctionSignatureGetNumInputArgs(MTRT_FunctionSignature signature,
                                     int64_t *numInputArgs) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numInputArgs = FunctionSignatureView(sig).getNumInputArgs();
  return mtrtStatusGetOk();
}

MTRT_Status
mtrtFunctionSignatureGetNumOutputArgs(MTRT_FunctionSignature signature,
                                      int64_t *numOutputArgs) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numOutputArgs = FunctionSignatureView(sig).getNumOutputArgs();
  return mtrtStatusGetOk();
}

MTRT_Status getTypeHelper(TypeUnionView typeUnionView, MTRT_Type *type) {
  // Allocate the TypeUnion object, populate it by moving in the
  // concrete object, and release it to be owned by the CAPI object.
  auto typeUnion = std::make_unique<mtrt::flat::TypeUnion>();
  // Extract the correct type.
  if (typeUnionView.isa<MemRefTypeView>()) {
    auto memrefView = typeUnionView.get<MemRefTypeView>();
    mtrt::flat::MemRefTypeT memref;
    memref.shape = memrefView.getShape();
    memref.strides = memrefView.getStrides();
    memref.element_type = memrefView.getElementType();
    memref.address_space = memrefView.getAddressSpace();
    typeUnion->Set(std::move(memref));
  }
  if (typeUnionView.isa<ScalarTypeView>()) {
    auto scalarView = typeUnionView.get<ScalarTypeView>();
    mtrt::flat::ScalarTypeT scalar;
    scalar.type = scalarView;
    typeUnion->Set(std::move(scalar));
  }
  *type = wrap(typeUnion.release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtFunctionSignatureGetResult(MTRT_FunctionSignature signature,
                                           int64_t index, MTRT_Type *type) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  auto typeUnionView = FunctionSignatureView(sig).getResult(index);
  return getTypeHelper(typeUnionView, type);
}

MTRT_Status mtrtFunctionSignatureGetArg(MTRT_FunctionSignature signature,
                                        int64_t index, MTRT_Type *type) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  auto typeUnionView = FunctionSignatureView(sig).getArg(index);
  return getTypeHelper(typeUnionView, type);
}

MTRT_Status
mtrtFunctionSignatureGetNumArgBounds(MTRT_FunctionSignature signature,
                                     int64_t *numArgBounds) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numArgBounds = FunctionSignatureView(sig).getNumArgBounds();
  return mtrtStatusGetOk();
}

MTRT_Status
mtrtFunctionSignatureGetNumResBounds(MTRT_FunctionSignature signature,
                                     int64_t *numResBounds) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  *numResBounds = FunctionSignatureView(sig).getNumResBounds();
  return mtrtStatusGetOk();
}

MTRT_Status
mtrtFunctionSignatureGetShapeFuncName(MTRT_FunctionSignature signature,
                                      MTRT_StringView *name) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  std::optional<std::string_view> str =
      FunctionSignatureView(sig).getShapeFunctionName();
  if (!str) {
    *name = mtrtStringViewCreate(nullptr, 0);
    return mtrtStatusGetOk();
  }
  *name = mtrtStringViewCreate(str->data(), str->size());
  return mtrtStatusGetOk();
}

MTRT_Status getBoundsHelper(BoundsUnionView boundsUnionView,
                            MTRT_Bounds *bounds) {
  // Allocate the BoundsUnion object, populate it by moving in the
  // concrete object, and release it to be owned by the CAPI object.
  auto boundsUnion = std::make_unique<mtrt::flat::BoundsUnion>();
  // Extract the correct type.
  if (boundsUnionView.isa<DimensionBoundsView>()) {
    auto dimBounds = boundsUnionView.get<DimensionBoundsView>();
    mtrt::flat::DimensionBoundsT dims;
    dims.min = dimBounds.getMin();
    dims.max = dimBounds.getMax();
    boundsUnion->Set(std::move(dims));
  }
  if (boundsUnionView.isa<ValueBoundsView>()) {
    auto valBounds = boundsUnionView.get<ValueBoundsView>();
    mtrt::flat::ValueBoundsT vals;
    vals.min = valBounds.getMin();
    vals.max = valBounds.getMax();
    boundsUnion->Set(std::move(vals));
  }
  *bounds = wrap(boundsUnion.release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtFunctionSignatureGetArgBound(MTRT_FunctionSignature signature,
                                             int64_t index,
                                             MTRT_Bounds *bounds) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  auto boundsUnionView = FunctionSignatureView(sig).getArgBound(index);
  return getBoundsHelper(boundsUnionView, bounds);
}

MTRT_Status
mtrtFunctionSignatureGetResultBound(MTRT_FunctionSignature signature,
                                    int64_t index, MTRT_Bounds *bounds) {
  const mtrt::flat::FunctionSignature *sig = unwrap(signature);
  auto boundsUnionView = FunctionSignatureView(sig).getResultBound(index);
  return getBoundsHelper(boundsUnionView, bounds);
}
