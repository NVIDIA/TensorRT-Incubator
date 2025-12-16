//===- kernel-translate.cpp -----------------------------------------------===//
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
///
/// Entrypoint for the translation driver.
///
//===----------------------------------------------------------------------===//
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir {
/// Register emitc-to-cpp translation. This has no header file so we declare it
/// here.
void registerToCppTranslation();
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerToCppTranslation();
  return failed(mlir::mlirTranslateMain(
      argc, argv, "MLIR-Kernel Translation Testing Tool"));
}
