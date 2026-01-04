//===- SerializeArtifacts.cpp ---------------------------------------------===//
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
/// Implementation of the `executor-serialize-artifacts` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Support/ArtifactManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "executor-serialize-artifacts"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORSERIALIZEARTIFACTSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

namespace {
static mtrt::compiler::ArtifactKind inferArtifactKindFromPath(StringRef path) {
  if (path.ends_with(".ptx"))
    return mtrt::compiler::ArtifactKind::PTXModule;
  if (path.ends_with(".trt_plan.bin") || path.ends_with(".trtengine"))
    return mtrt::compiler::ArtifactKind::TRTEngine;
  if (path.ends_with(".json"))
    return mtrt::compiler::ArtifactKind::Manifest;
  return mtrt::compiler::ArtifactKind::ConstantBlob;
}

static mtrt::compiler::ArtifactProducerInfo makeProducerInfo(StringRef pass,
                                                             Operation *op) {
  mtrt::compiler::ArtifactProducerInfo producer;
  producer.pass = pass.str();
  producer.opName = op ? op->getName().getStringRef().str() : "";
  producer.symbol = "";
  if (op) {
    llvm::raw_string_ostream os(producer.loc);
    op->getLoc().print(os);
    os.flush();
  } else {
    producer.loc = "";
  }
  return producer;
}

class ExecutorSerializeArtifactsPass
    : public executor::impl::ExecutorSerializeArtifactsPassBase<
          ExecutorSerializeArtifactsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *moduleOp = getOperation();

    // Get the artifact manager
    mtrt::compiler::ArtifactManager::Options options;
    if (!artifactsDirectory.empty())
      options.artifactsDirectory = artifactsDirectory;

    llvm::IntrusiveRefCntPtr<mtrt::compiler::ArtifactManager> artifactManager =
        llvm::makeIntrusiveRefCnt<mtrt::compiler::ArtifactManager>(options);

    LLVM_DEBUG(llvm::dbgs()
               << "Serializing artifacts to directory: "
               << artifactManager->getArtifactsDirectory() << "\n");

    auto dataLayout = DataLayout::closest(moduleOp);

    SmallVector<executor::FileArtifactOp> fileArtifactOps;
    const std::string passName = getArgument().str();

    for (auto fileOp :
         moduleOp->getRegion(0).getOps<executor::FileArtifactOp>()) {
      mtrt::compiler::ArtifactKind kind =
          inferArtifactKindFromPath(fileOp.getPath());
      mtrt::compiler::ArtifactProducerInfo producerInfo =
          makeProducerInfo(passName, fileOp.getOperation());
      llvm::Expected<mtrt::compiler::FinalizedArtifactRef>
          finalizedArtifactRef = artifactManager->addElementsAttr(
              fileOp.getPath(), fileOp.getDataAttr(), dataLayout, kind,
              producerInfo, fileOp->getLoc());
      if (!finalizedArtifactRef) {
        emitError(fileOp->getLoc())
            << "failed to add elements attr: "
            << llvm::toString(finalizedArtifactRef.takeError());
        return signalPassFailure();
      }
      fileArtifactOps.push_back(fileOp);
    }

    const bool hasArtifacts = !fileArtifactOps.empty();

    for (auto fileArtifactOp : fileArtifactOps)
      fileArtifactOp.erase();

    // Write the manifest if requested
    if (createManifest && hasArtifacts) {
      if (auto moduleOpBuiltin = dyn_cast<ModuleOp>(moduleOp)) {
        if (llvm::Error err = artifactManager->writeManifest(moduleOpBuiltin)) {
          emitError(moduleOp->getLoc()) << "failed to write artifact manifest: "
                                        << llvm::toString(std::move(err));
          return signalPassFailure();
        }
      }
    }

    // Keep all artifacts on teardown
    artifactManager->setShouldKeep(true);
  }
};
} // namespace
