//===- Compiler.h -------------------------------------------------*- C -*-===//
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
#include "mlir-executor-c/Target/ExecutorTranslations.h"
#include "mlir-executor-c/Support/Status.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir/CAPI/IR.h"

using namespace mlir;

MTRT_Status translateToRuntimeExecutable(MlirOperation op,
                                         MTRT_Executable *result) {
  FailureOr<std::unique_ptr<mlirtrt::runtime::ExecutableStorage>> exeStorage =
      mlir::translateToRuntimeExecutable(unwrap(op));
  if (failed(exeStorage))
    return mtrtStatusCreate(MTRT_StatusCode_InternalError,
                            "failed to translate to executable");

  *result = MTRT_Executable{
      std::make_unique<mlirtrt::runtime::Executable>(std::move(*exeStorage))
          .release()};

  return mtrtStatusGetOk();
}
