//===- Support.h ------------------------------------------------*- C++ -*-===//
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
/// Inclusions for TS-like items and template utilities. We re-use the
/// implementations from `llvm/ADT`, but this dependency could be replaced by
/// replacing with a local extentsion or another third party implementation.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_COMMON_SUPPORT_H
#define MLIR_TENSORRT_RUNTIME_BACKEND_COMMON_SUPPORT_H

#include "llvm/ADT/ScopeExit.h"

namespace mlirtrt {
using llvm::make_scope_exit;
}

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_COMMON_SUPPORT_H
