//===- RegionUtils.h --------------------------------------------*- C++ -*-===//
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
/// Utilities for common region operations.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace tensorrt {

/// For each used-value-defined-above within the single-block `body` region, if
/// the value is a scalar or vector-type constant, then clone that constant into
/// the region and use the clone instead of the above-defined constant within
/// the body. Return the remaining set of used values defined above but that
/// weren't cloned in.
SmallVector<Value>
sinkConstantsAndGetUsedValuesDefinedAbove(RewriterBase &rewriter, Region &body);

} // namespace tensorrt
} // namespace mlir
