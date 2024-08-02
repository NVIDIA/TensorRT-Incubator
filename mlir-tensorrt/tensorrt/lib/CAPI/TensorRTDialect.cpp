//===- Dialect.cpp --------------------------------------------------------===//
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
/// Implementation of TensorRT dialect registration using upstream C API.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect-c/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tensorrt;

#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TensorRT, tensorrt,
                                      mlir::tensorrt::TensorRTDialect)
