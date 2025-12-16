//===- RegisterAllDialects.h ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
///===- RegisterAllDialects.h -------------------------------------*- C -*-===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// C API registration functions for the Kernel dialects and passes and
/// required upstream dialects and interface implementations. These
/// registrations do not include dialects and dependencies of TensorRT, Plan,
/// or Executor dialects. These APIs are intended for use with clients that
/// are building code-generation tools.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_C_REGISTERALLDIALECTS
#define MLIR_KERNEL_C_REGISTERALLDIALECTS

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Add all the dialects used by MLIR-TensorRT for device code generation to the
/// registry.
MLIR_CAPI_EXPORTED void
mlirTensorRTRegisterCodegenDialects(MlirDialectRegistry registry);

/// Register all the compiler passes used by MLIR-TensorRT device code
/// generation.
MLIR_CAPI_EXPORTED void mlirTensorRTRegisterAllCodegenPasses();

#ifdef __cplusplus
}
#endif

#endif // MLIR_KERNEL_C_REGISTERALLDIALECTS
