//===- StaticValueUtils.h ---------------------------------------*- c++ -*-===//
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
// Utilities for dealing with constant values or attributes.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_STATICVALUEUTILS_H
#define MLIR_TENSORRT_UTILS_STATICVALUEUTILS_H

#include "mlir/IR/DialectResourceBlobManager.h" // IWYU pragma: keep
#include <optional>

namespace mlir {
class ElementsAttr;
/// If the `attr` is an elided DenseElementsAttr, then it will appear as a
/// DenseResourceElementsAttr with the handle "elided". If this is the case,
/// return the handle object, otherwise return nullopt.
std::optional<DenseResourceElementsHandle>
getElidedResourceElementsAttr(ElementsAttr attr);
} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_STATICVALUEUTILS_H
