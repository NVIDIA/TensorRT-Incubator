//===- TensorExtPadToInsertSlice.cpp --------------------------------------===//
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
/// This file implements a pass that converts `tensor.pad` operations to
/// `linalg.fill` and `tensor.insert_slice` operations.
///
/// Note: this pass  is only needed because the equivalent pass upstream is
/// restricted to run on `builtin.module`, but we want to run on functions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mtrt {
#define GEN_PASS_DEF_TENSOREXTPADTOINSERTSLICEPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;

namespace {
class TensorExtPadToInsertSlicePass
    : public mtrt::impl::TensorExtPadToInsertSlicePassBase<
          TensorExtPadToInsertSlicePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<linalg::DecomposePadOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
