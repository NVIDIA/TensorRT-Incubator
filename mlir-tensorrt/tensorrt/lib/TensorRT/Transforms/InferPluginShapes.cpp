//===- InferPluginShapes.cpp ----------------------------------------------===//
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
/// Implementation of the `tensorrt-infer-plugin-shapes` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/Utils/NvInferPluginUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tensorrt {
#define GEN_PASS_DEF_INFERPLUGINSHAPESPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace mlir::tensorrt

using namespace mlir;
using namespace mlir::tensorrt;

/// Updates the `pluginOp` by loading the plugin described by the op's metadata
/// and hijacking the plugin's shape expression machinery to generate MLIR IR in
/// the shape calculation region.
static LogicalResult
updatePluginShapeCalculationRegion(PluginManager &pluginManager,
                                   RewriterBase &rewriter,
                                   OpaquePluginOp pluginOp) {
  assert(pluginOp.getShapesRegion().empty() && "expected empty region");
  FailureOr<PluginInterfaceBase *> pluginBase = pluginManager.getExternalPlugin(
      pluginOp.getLoc(), pluginOp.getPluginName(), pluginOp.getPluginVersion(),
      pluginOp.getPluginNamespace(), pluginOp.getCreatorParams(), "no-name",
      pluginOp.getDsoPath(), {});

  if (failed(pluginBase))
    return pluginOp.emitOpError()
           << "failed to load pluginBase for shape inference";

  if (failed(buildPluginShapeRegion(
          pluginOp, *pluginBase,
          [](OpBuilder &b, Location loc, ArrayRef<Value> operands) {
            b.create<tensorrt::YieldOp>(loc, operands);
          })))
    return pluginOp.emitOpError() << "failed to construct shape region";

  return success();
}

namespace {
class InferPluginShapesPass
    : public tensorrt::impl::InferPluginShapesPassBase<InferPluginShapesPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    SmallVector<OpaquePluginOp> ops;
    op->walk([&](tensorrt::OpaquePluginOp plugin) {
      if (!llvm::all_of(plugin.getResultTypes(),
                        [](Type t) {
                          return cast<RankedTensorType>(t).hasStaticShape();
                        }) &&
          plugin->getRegion(0).empty()) {
        ops.push_back(plugin);
      } else {
      }
    });

    IRRewriter rewriter(ctx);
    PluginManager pluginManager;
    for (OpaquePluginOp pluginOp : ops) {
      if (failed(updatePluginShapeCalculationRegion(pluginManager, rewriter,
                                                    pluginOp)))
        return signalPassFailure();
    }
  }
};
} // namespace
