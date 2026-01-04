//===- Client.cpp ---------------------------------------------------------===//
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
/// Implementation for the CompilerClient.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "llvm/ADT/StringRef.h"

using namespace mtrt;
using namespace mtrt::compiler;
using namespace mlir;

#define DEBUG_TYPE "compiler-api"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<CompilerClient>>
CompilerClient::create(MLIRContext *context) {
  return std::unique_ptr<CompilerClient>(new CompilerClient(context));
}

CompilerClient::CompilerClient(mlir::MLIRContext *context) : context(context) {}

void CompilerClient::updateCachedPipeline(const llvm::hash_code &hashCode,
                                          std::unique_ptr<Pipeline> task) {
  cachedPassManagers[hashCode] = std::move(task);
}

Pipeline *
CompilerClient::lookupCachedPipeline(const llvm::hash_code &hashCode) const {
  auto it = cachedPassManagers.find(hashCode);
  if (it == cachedPassManagers.end())
    return nullptr;
  return it->second.get();
}

StatusOr<Pipeline *> CompilerClient::getOrCreatePipeline(
    llvm::IntrusiveRefCntPtr<MainOptions> options) {
  // Ensure options are in a consistent, validated state before hashing/caching.
  MTRT_RETURN_IF_ERROR(options->finalize());
  std::optional<llvm::hash_code> hashCode = options->getHash();
  if (!hashCode)
    return getInternalErrorStatus("failed to get hash code for options");
  Pipeline *pipeline = lookupCachedPipeline(*hashCode);
  if (pipeline)
    return pipeline;
  std::unique_ptr<Pipeline> newPipeline =
      std::make_unique<Pipeline>(context, std::move(options));
  pipeline = newPipeline.get();
  updateCachedPipeline(*hashCode, std::move(newPipeline));
  return pipeline;
}
