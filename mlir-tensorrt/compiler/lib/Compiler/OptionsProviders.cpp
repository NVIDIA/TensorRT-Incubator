//===- OptionsProviders.cpp -------------------------------------*- C++ -*-===//
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
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "cuda_runtime_api.h"
#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlirtrt;
using namespace mlirtrt::compiler;

//===----------------------------------------------------------------------===//
// DebugOptions
//===----------------------------------------------------------------------===//

void DebugOptions::applyToPassManager(PassManager &pm) const {
  // If the options specify to use global MLIR CL flags, then apply those
  // options. Otherwise, use our local options. Using global options is only
  // possible if the LLVM global command line flag environment is initialized
  // correctly.
  if (useGlobalCLPrintingOptions) {
    if (failed(applyPassManagerCLOptions(pm)))
      llvm::report_fatal_error("failed to populate pass manager "
                               "instrumentation from global CL options");
    applyDefaultTimingPassManagerCLOptions(pm);
    return;
  }

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

DeviceOptions::DeviceOptions(mlir::detail::PassOptions *ctx)
    : OptionsProvider(ctx) {
  shouldInferFromHost.setCallback([&](const bool &value) -> void {
    if (!value)
      return;
    StatusOr<DeviceInfo> deviceInfo = getDeviceInformationFromHost(0);
    if (!deviceInfo.isOk()) {
      llvm::report_fatal_error(deviceInfo.getString().c_str());
    }
    computeCapability = deviceInfo->computeCapability;
    maxRegistersPerBlock = deviceInfo->maxRegistersPerBlock;
    maxSharedMemoryPerBlockKb = deviceInfo->maxSharedMemoryPerBlockKb;
  });

  computeCapability.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      computeCapability = hostDeviceInfo->computeCapability;
    }
  });

  maxRegistersPerBlock.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      maxRegistersPerBlock = hostDeviceInfo->maxRegistersPerBlock;
    }
  });

  maxSharedMemoryPerBlockKb.setCallback([&](const int &value) -> void {
    if (hostDeviceInfo) {
      assert(shouldInferFromHost && "shouldInferFromHost must be true");
      maxSharedMemoryPerBlockKb = hostDeviceInfo->maxSharedMemoryPerBlockKb;
    }
  });
}

//===----------------------------------------------------------------------===//
// CompilationTaskOptionsBase
//===----------------------------------------------------------------------===//

std::optional<llvm::hash_code> CompilationTaskOptionsBase::getHash() const {
  // We hash by just hashing the string representation.
  llvm::SmallString<128> str;
  {
    llvm::raw_svector_ostream os(str);
    this->print(os);
  }
  return llvm::hash_value(str);
}

mlir::LogicalResult
CompilationTaskOptionsBase::parse(llvm::ArrayRef<llvm::StringRef> args,
                                  std::string &err) {
  std::string result;
  for (unsigned i = 0; i < args.size(); ++i) {
    llvm::StringRef part = args[i];
    while (part.starts_with("-"))
      part = part.drop_front(1);
    result += part;
    if (i < args.size() - 1)
      result += " ";
  }
  llvm::raw_string_ostream ss(err);
  if (failed(this->parseFromString(result, ss)))
    return mlir::failure();
  return mlir::success();
}
