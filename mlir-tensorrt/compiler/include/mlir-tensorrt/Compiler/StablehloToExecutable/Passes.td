//===- Passes.td ----------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// PopulateDefaultBackendMetadataPass
//===----------------------------------------------------------------------===//

def PopulateDefaultBackendMetadataPass : Pass<"populate-default-backend-metadata",
      "::mlir::ModuleOp"> {
  let summary = "Populate Plan dialect backend metadata information it not "
                "already present";

  let description = [{
    The compiler uses an attribute 'plan.backends' located on the module
    in order to understand the relative priority of each backend. If the
    attribute is not present, then this pass will populate it with a default
    value.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];

  let options = [
    ListOption<"backends", "backends", "std::string",
    "A list of default backend attributes to use for compilation if no "
    "'plan.backends' attribute is provided on the top-level module.",
    "llvm::cl::list_init<std::string>({\"#plan.host_backend<benefit=1>\"})">
  ];
}


//===----------------------------------------------------------------------===//
// ProcessStablehloHostClustersPass
//===----------------------------------------------------------------------===//

def ProcessStablehloHostClustersPass : Pass<"process-stablehlo-host-clusters",
      "::mlir::func::FuncOp"> {
  let summary = "Convert StableHLO inside host clusters into scalar arith/math operations";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertStablehloConstantsToArithPass
//===----------------------------------------------------------------------===//

def ConvertStablehloConstantsToArithPass :
      Pass<"convert-stablehlo-constants-to-arith", "::mlir::func::FuncOp"> {
  let summary = "Convert any 'stablehlo.constant' into 'arith.constant'";

  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES
