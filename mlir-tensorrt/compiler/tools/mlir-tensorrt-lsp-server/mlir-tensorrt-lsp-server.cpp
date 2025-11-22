//===- MlirTensorRtLspServer.cpp ------------------------------------------===//
//
// Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
/// This file is the entry point for the `mlir-tensorrt-lsp-server` executable.
/// This provides Language Server Protocol (LSP) for auto-complete and
/// diagnostics support to compatible IDEs (i.e. VSCode with MLIR extension).
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mtrt::compiler::registerAllDialects(registry);
  mtrt::compiler::registerAllExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
