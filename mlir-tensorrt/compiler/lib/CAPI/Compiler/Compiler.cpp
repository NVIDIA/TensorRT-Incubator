//===- Compiler.cpp -------------------------------------------------------===//
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
/// MLIR-TensorRT Compiler CAPI implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-c/Compiler/Compiler.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir-tensorrt-common-c/Support/Status.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Pass/PassManager.h"

using namespace mtrt;
using namespace mtrt::compiler;
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
/// the `mtrt::StatusCode`.
static MTRT_StatusCode
getMTRTStatusCodeFromRuntimeStatusCode(mtrt::StatusCode code) {
  return static_cast<MTRT_StatusCode>(code);
}

static MTRT_Status wrap(const mtrt::Status &status) {
  if (status.isOk())
    return mtrtStatusGetOk();
  return mtrtStatusCreate(
      getMTRTStatusCodeFromRuntimeStatusCode(status.getCode()),
      status.getMessage().c_str());
}

//===----------------------------------------------------------------------===//
// MTRT_CompilerClient
//===----------------------------------------------------------------------===//

MTRT_Status mtrtCompilerClientCreate(MlirContext context,
                                     MTRT_CompilerClient *client) {
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

MTRT_Status mtrtCompilerClientGetPipeline(MTRT_CompilerClient client,
                                          MlirStringRef taskMnemonic,
                                          const MlirStringRef *argv,
                                          unsigned argc,
                                          MlirPassManager *result) {
  std::vector<llvm::StringRef> argvStrRef(argc);
  for (unsigned i = 0; i < argc; i++)
    argvStrRef[i] = llvm::StringRef(argv[i].data, argv[i].length);
  StatusOr<PipelineBase *> pipeline = unwrap(client)->getPipeline(
      StringRef(taskMnemonic.data, taskMnemonic.length), argvStrRef,
      /*enableDebugOptions=*/true);
  if (!pipeline.isOk())
    return wrap(pipeline.getStatus());
  *result = MlirPassManager{static_cast<mlir::PassManager *>(*pipeline)};
  return mtrtStatusGetOk();
}
