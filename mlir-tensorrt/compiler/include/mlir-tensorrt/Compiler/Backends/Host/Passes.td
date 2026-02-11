//===- Passes.td ----------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_BACKENDS_HOST_PASSES
#define MLIR_TENSORRT_BACKENDS_HOST_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ProcessHostClustersPass
//===----------------------------------------------------------------------===//

def ProcessHostClustersPass : Pass<"process-host-clusters",
      "::mlir::func::FuncOp"> {
  let summary = "Convert StableHLO inside host clusters into scalar arith/math operations";

  let description = [{
    This pass executes a "convert-stablehlo-scalar-to-arith" dynamically on all
    functions with the #plan.host_backend target attribute. It converts StableHLO
    operations within host clusters into scalar arith/math operations suitable
    for CPU execution.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

#endif // MLIR_TENSORRT_BACKENDS_HOST_PASSES
