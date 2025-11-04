///===- SerializationUtils.cpp --------------------------------------------===//
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
#include "mlir-executor/Utils/SerializationUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

static LogicalResult
serializeSplatElementsAttr(SplatElementsAttr elAttr,
                           SerializationInterface &callback, uint64_t align) {
  if (elAttr.getElementType().isInteger(1)) {
    std::vector<int8_t> data(elAttr.getNumElements(),
                             elAttr.getSplatValue<bool>() ? 1 : 0);
    return callback.serialize(llvm::ArrayRef<int8_t>(data.data(), data.size()),
                              elAttr.getElementType(), align);
  }
  if (elAttr.getElementType().isInteger(4)) {
    std::vector<int8_t> data(
        elAttr.getNumElements(),
        static_cast<int8_t>(elAttr.getSplatValue<APInt>().getSExtValue()));
    return callback.serialize(llvm::ArrayRef<int8_t>(data.data(), data.size()),
                              elAttr.getElementType(), align);
  }
  ArrayRef<char> data = elAttr.getRawData();
  std::vector<int8_t> output;
  output.reserve(data.size() * elAttr.getNumElements());
  for (int64_t i = 0; i < elAttr.getNumElements(); i++)
    llvm::append_range(output, data);
  return callback.serialize(
      llvm::ArrayRef<int8_t>(output.data(), output.size()),
      elAttr.getElementType(), align);
}

static LogicalResult
serializeDenseElementsAttr(DenseIntOrFPElementsAttr attr,
                           SerializationInterface &callback, uint64_t align) {
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
    return serializeSplatElementsAttr(splatAttr, callback, align);
  if (attr.getElementType().isInteger(1)) {
    auto range = llvm::map_to_vector(attr.getValues<bool>(), [](bool inp) {
      return static_cast<int8_t>(inp);
    });
    return callback.serialize(
        llvm::ArrayRef<int8_t>(range.data(), range.size()),
        attr.getElementType(), align);
  }
  return callback.serialize(attr.getRawData(), attr.getElementType(), align);
}

static LogicalResult serializeDenseResourceElementsAttr(
    Location loc, DenseResourceElementsAttr resourceAttr,
    const DataLayout &dataLayout, SerializationInterface &callback,
    uint64_t align) {
  uint64_t sizeBytes = dataLayout.getTypeSize(resourceAttr.getElementType());
  DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
  if (!handle.getBlob())
    return emitError(loc, "resource blob does not exist for key: " +
                              handle.getKey());

  ArrayRef<char> data = handle.getBlob()->getData();
  if (data.size() != resourceAttr.getNumElements() * sizeBytes)
    return emitError(loc, "unexpected serialization size ")
           << data.size() << ", expected serialization size is "
           << resourceAttr.getNumElements() * sizeBytes << "\n";
  return callback.serialize(
      llvm::ArrayRef<int8_t>(reinterpret_cast<const int8_t *>(data.data()),
                             data.size()),
      resourceAttr.getElementType(), align);
}

LogicalResult mlir::serializeElementsAttr(Location loc, ElementsAttr attr,
                                          const DataLayout &dataLayout,
                                          SerializationInterface &callback,
                                          std::optional<uint64_t> alignment) {
  uint64_t align = dataLayout.getTypeABIAlignment(attr.getElementType());
  if (alignment) {
    assert(llvm::isPowerOf2_64(*alignment) && "Alignment must be a power of 2");
    align = std::max<uint64_t>(align, *alignment);
  }

  if (auto dense = dyn_cast<DenseIntOrFPElementsAttr>(attr))
    return serializeDenseElementsAttr(dense, callback, align);
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr))
    return serializeDenseResourceElementsAttr(loc, resource, dataLayout,
                                              callback, align);
  return emitError(loc) << "unsupported elements attribute type: " << attr;
}

FailureOr<uint64_t> mlir::getSerializedSize(Location loc, Attribute attr_,
                                            const DataLayout &dataLayout,
                                            std::optional<uint64_t> alignment) {
  if (auto attr = dyn_cast<ElementsAttr>(attr_)) {
    uint64_t sizeBytes = dataLayout.getTypeSize(attr.getElementType());
    uint64_t unalignedSize = sizeBytes * attr.getNumElements();
    uint64_t align = dataLayout.getTypeABIAlignment(attr.getElementType());
    if (alignment) {
      assert(llvm::isPowerOf2_64(*alignment) &&
             "Alignment must be a power of 2");
      align = std::max<uint64_t>(align, *alignment);
    }
    return llvm::alignTo(unalignedSize, align);
  }
  if (auto strAttr = dyn_cast<StringAttr>(attr_)) {
    return strAttr.strref().size();
  }
  return emitError(loc) << "unsupported attribute type: " << attr_;
}
