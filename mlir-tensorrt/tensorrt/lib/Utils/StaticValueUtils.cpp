//===- StaticValueUtils.cpp  ----------------------------------------------===//
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
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace mlir;

std::optional<DenseResourceElementsHandle>
mlir::getElidedResourceElementsAttr(ElementsAttr attr) {
  auto denseResourceAttr = dyn_cast<DenseResourceElementsAttr>(attr);
  if (!denseResourceAttr)
    return std::nullopt;
  DenseResourceElementsHandle handle = denseResourceAttr.getRawHandle();
  if (handle.getKey() != "__elided__")
    return std::nullopt;
  return handle;
}