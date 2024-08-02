//===- Passes.h- ------------------------------------------------*- c++ -*-===//
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
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES
#define MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES

#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <mlir/Pass/Pass.h>

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace tensorrt {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

namespace llvm::cl {
/// The LLVM CommandLine infrastructure parser for TensorRTVersion
template <>
class parser<mlir::tensorrt::TensorRTVersion>
    : public basic_parser<mlir::tensorrt::TensorRTVersion> {
public:
  parser(Option &o) : basic_parser(o) {}

  StringRef getValueName() const override {
    return "TensorRT version specifier";
  }

  bool parse(Option &o, StringRef ArgName, StringRef Arg,
             mlir::tensorrt::TensorRTVersion &Val);
  void printOptionDiff(const Option &O, mlir::tensorrt::TensorRTVersion,
                       const OptVal &Default, size_t GlobalWidth) const;
  static void print(raw_ostream &os,
                    const mlir::tensorrt::TensorRTVersion &value);
  void anchor() override {}
};

} // namespace llvm::cl

//===----------------------------------------------------------------------===//
// Pass Pipeline Declarations
//===----------------------------------------------------------------------===//

namespace mlir::tensorrt {
/// Buidls a standard pipeline `tensorrt-module-simplification-pipeline` for
/// applying simplifications/optimizations to TensorRT functions or modules.
/// These include things like broadcast and transpose elimination passes. This
/// pipeline is safe to run multiple times.
void buildTensorRTModuleSimplificationPipeline(OpPassManager &pm);

/// Build a standard pipeline `tensorrt-module-transformation-pipeline` for
/// optimizing and lowering `tensorrt` functions or modules containing tensorrt
/// functions. After this pipeline, functions can be translated to TensorRT
/// engines.
void buildTensorRTModuleTransformationPipeline(mlir::OpPassManager &pm,
                                               bool stronglyTyped);
} // namespace mlir::tensorrt

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES
