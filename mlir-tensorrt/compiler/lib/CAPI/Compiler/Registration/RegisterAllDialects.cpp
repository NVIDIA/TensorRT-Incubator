//===- Registration.cpp  --------------------------------------------------===//
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
/// Definitions of mlir-tensorrt compiler CAPI registration functions.
///
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt-c/Compiler/Registration/RegisterAllDialects.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-tensorrt/Features.h"
#include "mlir-tensorrt/InitAllDialects.h"
#include "mlir-tensorrt/InitAllExtensions.h"
#include "mlir-tensorrt/InitAllPasses.h"

#include "mlir/CAPI/IR.h"

void mtrtCompilerRegisterDialects(MlirDialectRegistry registry) {
  mlirtrt::compiler::registerAllDialects(*unwrap(registry));
  mlirtrt::compiler::registerAllExtensions(*unwrap(registry));
}

void mtrtCompilerRegisterPasses() { mlirtrt::compiler::registerAllPasses(); }

void mtrtCompilerRegisterTasks() {
  mlirtrt::compiler::registerStableHloToExecutableTask();
  mlirtrt::compiler::registerTensorRTToExecutableTask();
}
