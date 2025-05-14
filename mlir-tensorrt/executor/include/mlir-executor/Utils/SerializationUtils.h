///===- SerializationUtils.h -----------------------------------*- C++ -*-===//
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
/// Utilities for common region operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_UTILS_SERIALIZATION_UTILS_H
#define MLIR_EXECUTOR_UTILS_SERIALIZATION_UTILS_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir {

class SerializationInterface {
public:
  SerializationInterface(const DataLayout &dataLayout)
      : dataLayout(dataLayout) {}

  virtual ~SerializationInterface() {}

  virtual LogicalResult serialize(const char *data, size_t size,
                                  Type elementType, uint64_t align) = 0;

  template <typename T>
  LogicalResult serialize(ArrayRef<T> data, Type elementType,
                          std::optional<uint64_t> alignment) {
    uint64_t align = dataLayout.getTypeABIAlignment(elementType);
    if (alignment) {
      assert(llvm::isPowerOf2_64(*alignment) &&
             "Alignment must be a power of 2");
      align = std::max<uint64_t>(align, *alignment);
    }
    return serialize(reinterpret_cast<const char *>(data.data()), data.size(),
                     elementType, align);
  }

protected:
  const DataLayout &dataLayout;
};

/// Serialize an ElementsAttr using a SerializationInterface.
/// An optional alignment can be provided to override the default alignment
/// for the element type.
LogicalResult serializeElementsAttr(Location loc, ElementsAttr attr,
                                    const DataLayout &dataLayout,
                                    SerializationInterface &callback,
                                    std::optional<uint64_t> alignment = {});

/// Get the serialized size of an ElementsAttr.
/// An optional alignment can be provided to override the default alignment
/// for the element type.
FailureOr<uint64_t> getSerializedSize(Location loc, ElementsAttr attr,
                                      const DataLayout &dataLayout,
                                      std::optional<uint64_t> alignment = {});

} // namespace mlir

#endif // MLIR_EXECUTOR_UTILS_SERIALIZATION_UTILS_H
