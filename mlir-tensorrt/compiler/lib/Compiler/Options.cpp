//===- OptionsGroups.cpp -------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Data structures and functions for manipulating compiler options.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

llvm::cl::OptionCategory DebugOptions::category = {
    "MLIR-TensorRT Debug Options", ""};
llvm::cl::OptionCategory TensorRTOptions::category = {
    "MLIR-TensorRT Backend (TensorRT) Options", ""};
llvm::cl::OptionCategory ExecutorOptions::category = {
    "MLIR-TensorRT Executor (\"--host-target=executor\") Options", ""};
llvm::cl::OptionCategory EmitCOptions::category = {
    "MLIR-TensorRT EmitC (\"--host-target=emitc\") Options", ""};
llvm::cl::OptionCategory BufferizationOptions::category = {
    "MLIR-TensorRT Bufferization Options", ""};
llvm::cl::OptionCategory OptimizationOptions::category = {
    "MLIR-TensorRT Optimization Options", ""};
llvm::cl::OptionCategory KernelGenOptions::category = {
    "MLIR-TensorRT Backend (KernelGen) Options", ""};
llvm::cl::OptionCategory MainOptions::category = {
    "MLIR-TensorRT Compilation Options", ""};
llvm::cl::OptionCategory AsyncSchedulingOptions::category = {
    "MLIR-TensorRT Async Scheduling Options", ""};
llvm::cl::OptionCategory DeviceOptions::category = {"CUDA Device Options", ""};

//===----------------------------------------------------------------------===//
// DebugOptions
//===----------------------------------------------------------------------===//

DebugOptions::DebugOptions(mlir::CLOptionScope &ctx) : OptionsGroup(ctx) {
  // Assert that the OptionsSet has a local scope.
  assert(!this->ctx.isGlobalScope() &&
         "DebugOptions must be constructed with a local scope");
}

void DebugOptions::applyToPassManager(PassManager &pm) const {
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  // Enable statistics dumping.
  if (passStatistics)
    pm.enableStatistics(mlir::PassDisplayMode::Pipeline);

  // Generate a reproducer on crash/failure.
  if (!reproducerFile.empty())
    pm.enableCrashReproducerGeneration(reproducerFile, localReproducer);

  if (enableTiming) {
    auto tm = std::make_unique<DefaultTimingManager>();
    tm->setEnabled(true);
    tm->setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
    pm.enableTiming(std::move(tm));
  }

  // Handle print-before.
  if (printBeforeAll) {
    // If we are printing before all, then just return true for the filter.
    shouldPrintBeforePass = [](Pass *, Operation *) { return true; };
  }
  // Handle print-after.
  if (printAfterAll || printAfterFailure) {
    // If we are printing after all or failure, then just return true for the
    // filter.
    shouldPrintAfterPass = [](Pass *, Operation *) { return true; };
  }

  // If there are no valid printing filters, then just return.
  if (!shouldPrintBeforePass && !shouldPrintAfterPass)
    return;

  OpPrintingFlags printFlags{};
  if (this->elideElementsAttrIfLarger > 0)
    printFlags.elideLargeElementsAttrs(this->elideElementsAttrIfLarger);
  if (this->elideResourceStringsIfLarger > 0)
    printFlags.elideLargeResourceString(this->elideResourceStringsIfLarger);

  // Otherwise, add the IR printing instrumentation.
  if (!printTreeDir.empty()) {
    pm.enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                  printModuleScope, printAfterChange,
                                  printAfterFailure, printTreeDir, printFlags);
    return;
  }

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printModuleScope, printAfterChange, printAfterFailure,
                      llvm::errs(), printFlags);
}

//===----------------------------------------------------------------------===//
// DeviceOptions
//===----------------------------------------------------------------------===//

DeviceOptions::DeviceOptions(mlir::CLOptionScope &ctx) : OptionsGroup(ctx) {
  shouldInferFromHost.setCallback([&](const bool &value) -> void {
    if (!value)
      return;
    StatusOr<DeviceInfo> deviceInfo = getDeviceInformationFromHost(0);
    mtrt::cantFail(deviceInfo);
    hostDeviceInfo = *deviceInfo;

    computeCapability.setValue(deviceInfo->computeCapability);
    maxRegistersPerBlock.setValue(deviceInfo->maxRegistersPerBlock);
    maxSharedMemoryPerBlockKb.setValue(deviceInfo->maxSharedMemoryPerBlockKb);
  });

  computeCapability.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      computeCapability.setValue(hostDeviceInfo->computeCapability);
    }
  });

  maxRegistersPerBlock.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      maxRegistersPerBlock.setValue(hostDeviceInfo->maxRegistersPerBlock);
    }
  });

  maxSharedMemoryPerBlockKb.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      maxSharedMemoryPerBlockKb.setValue(
          hostDeviceInfo->maxSharedMemoryPerBlockKb);
    }
  });
}

//===----------------------------------------------------------------------===//
// MainOptions
//===----------------------------------------------------------------------===//

Status MainOptions::validate() const { return getOkStatus(); }

template <typename OptionType, typename ValueType>
void updateIfUnset(OptionType &option, ValueType value) {
  if (option.getNumOccurrences() == 0) {
    option.setValue(value);
  }
}

Status MainOptions::finalize() {
  // Finalization is intentionally idempotent: callers may finalize explicitly
  // and pipeline construction may also finalize defensively.
  if (finalized)
    return getOkStatus();

  finalized = true;
  auto &bufferizationOpts = get<BufferizationOptions>();
  auto &clusteringOpts = get<plan::PlanClusteringOptions>();
  if (inputKind == plan::InputKind::TensorRT) {
    // For the TensorRT input kind, we default to using the
    // callee-allocating calling convention for entrypoint functions. Only
    // update the value if the user didn't explicitly force it to a specific
    // value.
    updateIfUnset(bufferizationOpts.forceEntrypointsReturnAllocs, true);
    updateIfUnset(clusteringOpts.preferAllocCallingConvention, true);
  }

  return validate();
}

std::optional<llvm::hash_code> MainOptions::getHash() const {
  // We hash by just hashing the string representation.
  llvm::SmallString<128> str;
  {
    llvm::raw_svector_ostream os(str);
    this->print(os);
  }
  return llvm::hash_value(str);
}

StatusOr<llvm::IntrusiveRefCntPtr<MainOptions>>
MainOptions::fromString(llvm::StringRef optionString,
                        ExtensionList extensions) {
  std::string errorMessage;
  mlir::CLOptionScope::ErrorCallback callback = [&](llvm::StringRef message) {
    errorMessage = message.str();
  };
  auto options = llvm::makeIntrusiveRefCnt<MainOptions>(
      mlir::CLOptionScope::LocalScope{}, std::move(extensions));
  if (failed(options->parseFromString(optionString, callback)))
    return getInternalErrorStatus(errorMessage.c_str());

  return options;
}
