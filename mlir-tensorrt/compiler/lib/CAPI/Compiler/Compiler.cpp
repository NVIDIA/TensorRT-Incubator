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
#include "mlir-c/Support.h"
#include "mlir-executor-c/Support/Status.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/StableHloToExecutable.h"
#include "mlir-tensorrt/Compiler/TensorRTExtension/TensorRTExtension.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/CAPI/IR.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
DEFINE_C_API_PTR_METHODS(MTRT_CompilerClient, CompilerClient)
DEFINE_C_API_PTR_METHODS(MTRT_StableHLOToExecutableOptions,
                         StableHLOToExecutableOptions)
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
             compiler::StableHloToExecutableTask,
             compiler::StableHLOToExecutableTensorRTExtension>()))
    emitWarning(mlir::UnknownLoc::get(ctx))
        << "ignoring duplicate extension load request; TensorRTExtension is "
           "already loaded";

  StatusOr<std::unique_ptr<CompilerClient>> cppClient =
      CompilerClient::create(unwrap(context));
  if (!cppClient.isOk())
    return wrap(cppClient.getStatus());

  *client = MTRT_CompilerClient{cppClient->release()};
  return mtrtStatusGetOk();
}

MTRT_Status mtrtCompilerClientDestroy(MTRT_CompilerClient client) {
  delete reinterpret_cast<MTRT_CompilerClient *>(client.ptr);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

MTRT_Status mtrtStableHloToExecutableOptionsCreate(
    MTRT_CompilerClient client, MTRT_StableHLOToExecutableOptions *options,
    int32_t tensorRTBuilderOptLevel, bool tensorRTStronglyTyped) {

  tensorrt::TensorRTTranslationOptions translationOpts;
  translationOpts.tensorrtBuilderOptLevel = tensorRTBuilderOptLevel;
  translationOpts.enableStronglyTyped = tensorRTStronglyTyped;

  // Load available extensions.
  MLIRContext *context = unwrap(client)->getContext();
  mlir::plan::PlanDialect *planDialect =
      context->getLoadedDialect<mlir::plan::PlanDialect>();
  compiler::TaskExtensionRegistry extensions =
      planDialect->extensionConstructors
          .getExtensionRegistryForTask<compiler::StableHloToExecutableTask>();

  // Check that default extension set is loaded and set options on the TRT
  // extension.
  auto *trtExtension =
      extensions
          .getExtension<compiler::StableHLOToExecutableTensorRTExtension>();
  assert(trtExtension &&
         "expected valid StableHLOToExecutableTensorRTExtension");
  trtExtension->setOptions(translationOpts);

  auto result =
      std::make_unique<StableHLOToExecutableOptions>(std::move(extensions));

  /// Populate device options from host information.
  Status inferStatus = result->inferDeviceOptionsFromHost();
  if (!inferStatus.isOk())
    return wrap(inferStatus);

  *options = wrap(result.release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStableHloToExecutableOptionsCreateFromArgs(
    MTRT_CompilerClient client, MTRT_StableHLOToExecutableOptions *options,
    const MlirStringRef *argv, unsigned argc) {

  // Load available extensions.
  MLIRContext *context = unwrap(client)->getContext();
  mlir::plan::PlanDialect *planDialect =
      context->getLoadedDialect<mlir::plan::PlanDialect>();
  compiler::TaskExtensionRegistry extensions =
      planDialect->extensionConstructors
          .getExtensionRegistryForTask<compiler::StableHloToExecutableTask>();

  // Check that default extension set is loaded.
  assert(
      extensions
          .getExtension<compiler::StableHLOToExecutableTensorRTExtension>() &&
      "expected valid StableHLOToExecutableTensorRTExtension");

  auto result =
      std::make_unique<StableHLOToExecutableOptions>(std::move(extensions));
  std::vector<llvm::StringRef> argvStrRef(argc);
  for (unsigned i = 0; i < argc; i++)
    argvStrRef[i] = llvm::StringRef(argv[i].data, argv[i].length);

  std::string err;
  if (failed(result->parse(argvStrRef, err))) {
    std::string line = llvm::join(argvStrRef, " ");
    return wrap(getInternalErrorStatus(
        "failed to parse options string {0} due to error: {1}", line, err));
  }

  /// Populate device options from host information.
  Status inferStatus = result->inferDeviceOptionsFromHost();
  if (!inferStatus.isOk())
    return wrap(inferStatus);

  *options = wrap(result.release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStableHloToExecutableOptionsSetDebugOptions(
    MTRT_StableHLOToExecutableOptions options, bool enableDebugging,
    const char **debugTypes, size_t debugTypeSizes, const char *dumpIrTreeDir,
    const char *dumpTensorRTDir) {

  StableHLOToExecutableOptions *cppOpts = unwrap(options);
  cppOpts->debugOptions.enableLLVMDebugFlag = enableDebugging;
  for (unsigned i = 0; i < debugTypeSizes; i++)
    cppOpts->debugOptions.llvmDebugTypes.emplace_back(debugTypes[i]);

  if (dumpIrTreeDir)
    cppOpts->debugOptions.dumpIRPath = std::string(dumpIrTreeDir);

  return mtrtStatusGetOk();
}

MTRT_Status
mtrtStableHloToExecutableOptionsSetTensorRTTranslationMetadataCallback(
    MTRT_StableHLOToExecutableOptions options, MTRT_MetadataCallback callback,
    void *userData) {
  StableHLOToExecutableOptions *cppOpts = unwrap(options);

  // Construct the append callback which we will pass to the callback provided
  // by the user. We do it this way to avoid needing a string construct in the C
  // API.
  auto appendFunc = [](MlirStringRef str, void *appendCtx) {
    std::string &accum = *reinterpret_cast<std::string *>(appendCtx);
    accum += std::string(str.data, str.length);
  };

  // Capturing by reference here will cause `callback` to point to the wrong
  // place at the time this callback is invoked.
  cppOpts->layerMetadataCallback = [=](Operation *op) {
    std::string accum;
    void *appendCtx = reinterpret_cast<void *>(&accum);
    callback(wrap(op), appendFunc, appendCtx, userData);
    return accum;
  };

  return mtrtStatusGetOk();
}

MTRT_Status mtrtStableHloToExecutableOptionsDestroy(
    MTRT_StableHLOToExecutableOptions options) {
  delete reinterpret_cast<StableHLOToExecutableOptions *>(options.ptr);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// Main StableHLO Compiler API Functions
//===----------------------------------------------------------------------===//

MTRT_Status mtrtCompilerStableHLOToExecutable(
    MTRT_CompilerClient client, MlirOperation module,
    MTRT_StableHLOToExecutableOptions stableHloToExecutableOptions,
    MTRT_Executable *result) {
  ModuleOp moduleOp = llvm::dyn_cast<ModuleOp>(unwrap(module));
  if (!moduleOp)
    return mtrtStatusCreate(
        MTRT_StatusCode::MTRT_StatusCode_InvalidArgument,
        "StableHLO-to-Executable compilation expects a ModuleOp");

  StatusOr<std::unique_ptr<mlirtrt::runtime::Executable>> exe =
      compiler::StableHloToExecutableTask::compileStableHLOToExecutable(
          *unwrap(client), moduleOp, *unwrap(stableHloToExecutableOptions));
  if (!exe.isOk())
    return mtrtStatusCreate(MTRT_StatusCode::MTRT_StatusCode_InternalError,
                            exe.getString().c_str());

  result->ptr = (*exe).release();

  return mtrtStatusGetOk();
}
