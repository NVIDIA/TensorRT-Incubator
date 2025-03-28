//===- TensorRTBackend.h ---------------------------------------*- C++ -*-===//
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
/// This file contains the declarations for the TensorRTBackend extensions to
/// the Plan dialect. It defines the TensorRTClusterKindAttr; see the associated
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND
#define MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND

#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackendAttrs.h.inc"

namespace mlir::plan {

/// Register the TensorRT backend extensions to the Plan dialect.
void registerTensorRTBackend(DialectRegistry &registry);

} // namespace mlir::plan

#endif // MLIR_TENSORRT_BACKENDS_TENSORRT_TENSORRTBACKEND
