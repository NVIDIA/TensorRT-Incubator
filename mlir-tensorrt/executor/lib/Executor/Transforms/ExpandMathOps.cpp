//===- ExpandMathOps.cpp --------------------------------------------------===//
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
/// Implementation of the `executor-expand-math-ops` pass.
///
/// This pass expands Math dialect operations that are not directly supported
/// by the Executor dialect into compositions of supported operations using
/// patterns from the upstream MLIR Math dialect transforms.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"              // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"                // IWYU pragma: keep
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTOREXPANDMATHOPSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

namespace {

/// Returns true if the given Math operation is NOT directly supported by the
/// Executor dialect and should be expanded/approximated.
///
/// Operations that ARE supported by Executor (from ExecutorOps.td):
/// - Unary float: absf, cbrt, ceil, cos, erf, exp, exp2, expm1, floor, log,
///                log10, log1p, log2, negf, sin, sqrt, tan, tanh, round
/// - Binary float: atan2, copysign
/// - Integer: absi
///
/// These should NOT be expanded since they have direct Executor equivalents.
static bool shouldExpandMathOp(StringRef opName) {
  // Operations directly supported by Executor dialect - do NOT expand these.
  static const llvm::StringSet<> executorSupportedOps = {
      // Unary float ops (from Executor_ArithUnaryMathFloatOp)
      "math.absf",
      "math.cbrt",
      "math.ceil",
      "math.cos",
      "math.erf",
      "math.exp",
      "math.exp2",
      "math.expm1",
      "math.floor",
      "math.log",
      "math.log10",
      "math.log1p",
      "math.log2",
      "math.sin",
      "math.sqrt",
      "math.tan",
      "math.tanh",
      "math.round",
      // Binary float ops (from Executor_ArithBinaryMathFloatOp)
      "math.atan2",
      "math.copysign",
      // Integer ops
      "math.absi",
      "math.ctpop",
  };

  return !executorSupportedOps.contains(opName);
}

class ExecutorExpandMathOpsPass
    : public executor::impl::ExecutorExpandMathOpsPassBase<
          ExecutorExpandMathOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // Add expansion patterns for Math operations that can be decomposed
    // into more primitive operations. These patterns rewrite operations like:
    // - sinh(x) -> (exp(x) - exp(-x)) / 2
    // - cosh(x) -> (exp(x) + exp(-x)) / 2
    // - asinh, acosh, atanh -> compositions of log, sqrt, etc.
    // - powf(x, y) -> exp(y * log(x))
    // - fpowi -> iterative multiplication
    // - rsqrt(x) -> 1 / sqrt(x)
    // - fma(a, b, c) -> a * b + c
    // - clampf(x, min, max) -> max(min(x, max), min)
    // - roundeven, ctlz, etc.
    //
    // We filter to only expand operations NOT supported by Executor.
    // Operations like tan, tanh, ceil, exp2, round are supported by Executor
    // and should NOT be expanded.
    SmallVector<StringRef> opsToExpand = {
        "sinh",      // -> (exp(x) - exp(-x)) / 2
        "cosh",      // -> (exp(x) + exp(-x)) / 2
        "asinh",     // -> log(x + sqrt(x^2 + 1))
        "acosh",     // -> log(x + sqrt(x^2 - 1))
        "atanh",     // -> 0.5 * log((1 + x) / (1 - x))
        "fma",       // -> a * b + c
        "powf",      // -> exp(y * log(x))
        "fpowi",     // -> iterative multiplication
        "rsqrt",     // -> 1 / sqrt(x)
        "clampf",    // -> max(min(x, max), min)
        "roundeven", // -> rounding with ties to even
        "ctlz",      // -> count leading zeros expansion
    };
    mlir::math::populateExpansionPatterns(patterns, opsToExpand);

    // Add patterns to promote low-precision float types (f16, bf16, f8) to f32
    // around math operations that need polynomial approximation. This wraps
    // the operation with arith.extf (to f32) and arith.truncf (back to original
    // type), allowing the approximation patterns to work on f32.
    //
    // We only promote operations that are NOT supported by Executor and need
    // approximation (e.g., atan, asin, acos, erfc).
    mlir::populateMathF32ExpansionPatterns(patterns, shouldExpandMathOp);

    // Add polynomial approximation patterns for operations that cannot be
    // simply expanded. We only enable approximations for operations NOT
    // supported by the Executor dialect.
    //
    // Operations that need approximation:
    // - atan -> polynomial approximation (atan2 is supported, but atan is not)
    // - asin, acos -> polynomial approximation based on atan
    // - erfc -> polynomial approximation (erf is supported, but erfc is not)
    //
    // We explicitly do NOT approximate operations that Executor supports
    // natively (sin, cos, exp, log, tanh, etc.) since the native versions
    // will be more accurate and potentially faster.
    mlir::populateMathPolynomialApproximationPatterns(patterns,
                                                      shouldExpandMathOp);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc(),
                "failed to apply math expansion patterns in ")
          << getArgument();
      return signalPassFailure();
    }
  }
};

} // namespace
