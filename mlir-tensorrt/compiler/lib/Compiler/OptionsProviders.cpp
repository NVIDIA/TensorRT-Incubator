//===- OptionsProviders.cpp -------------------------------------*- C++ -*-===//
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
/// Data structures and functions for manipulating compiler options.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "cuda_runtime_api.h"
#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/Error.h"

using namespace mlir;
using namespace mlirtrt;
using namespace mlirtrt::compiler;

//===----------------------------------------------------------------------===//
// DebugOptions
//===----------------------------------------------------------------------===//

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

  // Otherwise, add the IR printing instrumentation.
  if (!printTreeDir.empty()) {
    pm.enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                  printModuleScope, printAfterChange,
                                  printAfterFailure, printTreeDir);
    return;
  }

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printModuleScope, printAfterChange, printAfterFailure,
                      llvm::errs());
}

//===----------------------------------------------------------------------===//
// DeviceOptions
//===----------------------------------------------------------------------===//

Status DeviceOptions::inferFromHost() {
  cudaDeviceProp properties;
  cudaError_t err = cudaGetDeviceProperties(&properties, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device properties");
  int ccMajor = 0;
  int ccMinor = 0;
  err = cudaDeviceGetAttribute(
      &ccMajor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device compute capability");
  err = cudaDeviceGetAttribute(
      &ccMinor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device compute capability");
  // We want SM version as a single number.
  int64_t smVersion = ccMajor * 10 + ccMinor;
  info.computeCapability = smVersion;
  info.maxSharedMemoryPerBlockKb = properties.sharedMemPerBlock / 1024;
  info.maxRegistersPerBlock = properties.regsPerBlock;
  return Status::getOk();
}

llvm::Error DeviceOptions::finalizeImpl() {
  if (shouldInferFromHost) {
    StatusOr<DeviceInfo> deviceInfo = getDeviceInformationFromHost();

    if (!deviceInfo.isOk())
      return llvm::createStringError(deviceInfo.getString());

    info = *deviceInfo;
  }
  return llvm::Error::success();
}
