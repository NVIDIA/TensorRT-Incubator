//===- Compiler.cpp -------------------------------------------------------===//
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
/// MLIR-TensorRT Compiler CAPI implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-c/Compiler/Compiler.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir-executor-c/Support/Status.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
DEFINE_C_API_PTR_METHODS(MTRT_CompilerClient, CompilerClient)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

/// Return the MTRT_StatusCode. These are auto-generated from the same schema as
/// the `mlirtrt::StatusCode`.
static MTRT_StatusCode
getMTRTStatusCodeFromRuntimeStatusCode(mlirtrt::StatusCode code) {
  return static_cast<MTRT_StatusCode>(code);
}

static MTRT_Status wrap(const mlirtrt::Status &status) {
  if (status.isOk())
    return mtrtStatusGetOk();
  return mtrtStatusCreate(
      getMTRTStatusCodeFromRuntimeStatusCode(status.getCode()),
      status.getString().c_str());
}

//===----------------------------------------------------------------------===//
// MTRT_CompilerClient
//===----------------------------------------------------------------------===//

MTRT_Status mtrtCompilerClientCreate(MlirContext context,
                                     MTRT_CompilerClient *client) {
  MLIRContext *ctx = unwrap(context);
  // Populate default task extension set. This assumes the PlanDialect has
  // already been loaded.
  // TODO: We should only modify the loaded PlanDialect
  // via a class derived from DialectExtension and using
  // `registry.addExtensions` to append to the registry.
  mlir::plan::PlanDialect *planDialect =
      ctx->getOrLoadDialect<mlir::plan::PlanDialect>();
  assert(planDialect && "expected loaded PlanDialect");
  if (failed(planDialect->extensionConstructors.addCheckedExtensionConstructor<
             compiler::StablehloToExecutableTask,
             compiler::StablehloToExecutableTensorRTExtension>()))
    emitWarning(mlir::UnknownLoc::get(ctx))
        << "ignoring duplicate extension load request; TensorRTExtension is "
           "already loaded";

  StatusOr<std::unique_ptr<CompilerClient>> cppClient =
      CompilerClient::create(unwrap(context));
  if (!cppClient.isOk())
    return wrap(cppClient.getStatus());

  *client = wrap(cppClient->release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtCompilerClientDestroy(MTRT_CompilerClient client) {
  delete unwrap(client);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtCompilerClientGetCompilationTask(MTRT_CompilerClient client,
                                                 MlirStringRef taskMnemonic,
                                                 const MlirStringRef *argv,
                                                 unsigned argc,
                                                 MlirPassManager *result) {
  std::vector<llvm::StringRef> argvStrRef(argc);
  for (unsigned i = 0; i < argc; i++)
    argvStrRef[i] = llvm::StringRef(argv[i].data, argv[i].length);
  StatusOr<CompilationTaskBase *> task = unwrap(client)->getCompilationTask(
      StringRef(taskMnemonic.data, taskMnemonic.length), argvStrRef,
      /*enableDebugOptions=*/true);
  if (!task.isOk())
    return wrap(task.getStatus());
  *result = MlirPassManager{static_cast<mlir::PassManager *>(*task)};
  return mtrtStatusGetOk();
}
