//===- BoundsAttrInterface.h -----------------------------------*- C++ -*-===//
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
/// This file contains the declarations for the `BoundsAttrInterface` interface.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
struct FloatValueRange {
  APFloat min;
  APFloat max;
};
} // namespace mlir

#include "mlir-tensorrt-common/Interfaces/BoundsAttrInterface.h.inc"

#endif // MLIR_TENSORRT_COMMON_INTERFACES_BOUNDSATTRINTERFACE
