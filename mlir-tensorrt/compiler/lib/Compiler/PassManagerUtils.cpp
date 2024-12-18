//===- PassManagerUtils.cpp -------------------------------------*- C++ -*-===//
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

#include "mlir-tensorrt/Compiler/PassManagerUtils.h"

using namespace mlirtrt::compiler;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

mlir::LogicalResult setupPassManager(mlir::PassManager &pm,
                                     const DebugOptions &options) {
  pm.enableVerifier(true);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(mlir::applyPassManagerCLOptions(pm)))
    return mlir::failure();
  if (!options.dumpIRPath.empty()) {
    pm.enableIRPrintingToFileTree(
        [](Pass *, Operation *) { return false; },
        [](Pass *, Operation *) { return true; }, true, false, false,
        options.dumpIRPath, OpPrintingFlags().elideLargeElementsAttrs(32));
  }
  return mlir::success();
}
