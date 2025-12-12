//===- API.h --------------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_INTEGRATIONS_PJRT_CAPI_API
#define MLIR_TENSORRT_INTEGRATIONS_PJRT_CAPI_API

#include "mlir_trt_pjrt_export.h"
#include "xla/pjrt/c/pjrt_c_api.h"

//===----------------------------------------------------------------------===//
// This is the section for defining exported symbols.
// These symbols will be dynamically loaded from the `.so` at runtime
// by the XLA runtime (PJRT).
//===----------------------------------------------------------------------===//
#ifdef __cplusplus
extern "C" {
#endif

/// The primary object that implements the PJRT API is
/// the PJRT_Api struct. Our plugin will return an instance of this
/// struct to PJRT via `GetPjrtApi`.
///
/// The struct contains a number of function pointers that must
/// be implemented by the plugin.
/// See `tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h` for a complete list.
///
/// Conceptually speaking, the interface we need to implement is quite
/// simple. The functions of the interface are grouped into the APIs:
///
///   1. Error Handling
///   2. Event Management
///   3. Client Management
///   4. Device Mangement
///   5. Executable Management
///   6. Buffer Management

/// Allocate and return an initialized PJRT_Api struct. This memory is owned
/// by us (the plugin) and will not be deinitialized by the framework.
/// In fact, the framework sees this as `const PJRT_Api*`.
/// NOTE: Does not pass ownership of returned PJRT_Api* to caller.
MLIR_TRT_PJRT_EXPORT const PJRT_Api *GetPjrtApi();

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_INTEGRATIONS_PJRT_CAPI_API
