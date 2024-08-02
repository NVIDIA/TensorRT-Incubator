//===- TensorRTEncodingOpInterface.h ----------------------------*- C++ -*-===//
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
/// Declartion of the interface for operations that can be encoded into a
/// TensorRT nvinfer1::INetworkDefinition.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTENCODINGOPINTERFACE_H
#define MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTENCODINGOPINTERFACE_H

#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/NetworkEncoder.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"

namespace mlir {
namespace tensorrt {}
} // namespace mlir

#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h.inc"

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_TARGET_TENSORRTENCODINGOPINTERFACE_H
