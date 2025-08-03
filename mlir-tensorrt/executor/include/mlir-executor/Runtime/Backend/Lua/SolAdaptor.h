//===- SolAdaptor.h ---------------------------------------------*- C++ -*-===//
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
/// Wraps sol2 includes with diagnostic suppression pragmas.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_LUA_SOLADAPTOR
#define MLIR_TENSORRT_RUNTIME_BACKEND_LUA_SOLADAPTOR

#if defined(__clang__)
#pragma GCC diagnostic push
// In Debug builds, Sol2 has an ASSERT macro that is problematic due to this
// string conversion.
#pragma GCC diagnostic ignored "-Wstring-conversion"
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#define SOL_ALL_SAFETIES_ON 1
#include "sol/sol.hpp"
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_LUA_SOLADAPTOR
