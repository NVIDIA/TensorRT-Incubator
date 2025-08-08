//===- RaiseActivations.cpp  ----------------------------------------------===//
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
/// Implementation of `tensorrt-raise-activations`.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tensorrt {
#define GEN_PASS_DEF_RAISEACTIVATIONSPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace mlir::tensorrt

using namespace mlir;
using namespace mlir::tensorrt;

#include "RaiseActivations.pdll.h.inc"

namespace {
class RaiseActivations
    : public tensorrt::impl::RaiseActivationsPassBase<RaiseActivations> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    if (targetTensorRTVersion >= TensorRTVersion(10, 0)) {
      patterns.add<RaiseToGeluTanh>(ctx);
      patterns.add<RaiseToGeluTanh2>(ctx);
      patterns.add<RaiseToGeluErf>(ctx);
    }
    patterns.add<RaiseMaxMinToClip>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
