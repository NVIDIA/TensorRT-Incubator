//===- Passes.cpp ---------- ----------------------------------------------===//
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
/// Implementation of TensorRT pass pipelines and utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::tensorrt;

//===----------------------------------------------------------------------===//
// LLVM CL Adaptors
//===----------------------------------------------------------------------===//
bool llvm::cl::parser<tensorrt::TensorRTVersion>::parse(
    llvm::cl::Option &o, StringRef argName, StringRef argValue,
    tensorrt::TensorRTVersion &val) {
  SmallVector<int64_t> parts;
  for (StringRef part : llvm::split(argValue, ".")) {
    parts.emplace_back();
    if (part.getAsInteger(10, parts.back()))
      return o.error(Twine("invalid version: ") + argValue);
  }
  if (parts.size() > 4)
    return o.error(Twine("invalid version: ") + argValue);

  parts.resize(4, 0);
  val = TensorRTVersion(parts[0], parts[1], parts[2], parts[3]);
  return false;
}

void llvm::cl::parser<tensorrt::TensorRTVersion>::printOptionDiff(
    const Option &opt, tensorrt::TensorRTVersion value,
    const OptVal &defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= "
         << llvm::formatv("{0}.{1}.{2}.{3}", value.major, value.minor,
                          value.patch, value.build);
  if (defaultValue.hasValue()) {
    const TensorRTVersion &v = defaultValue.getValue();
    outs().indent(2) << " (default: ";
    outs() << llvm::formatv("{0}.{1}.{2}.{3}", v.major, v.minor, v.patch,
                            v.build)
           << ")";
  }
  outs() << "\n";
}

void llvm::cl::parser<tensorrt::TensorRTVersion>::print(
    raw_ostream &os, const tensorrt::TensorRTVersion &value) {
  os << llvm::formatv("{0}.{1}.{2}.{3}", value.major, value.minor, value.patch,
                      value.build);
}

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void tensorrt::buildTensorRTModuleSimplificationPipeline(OpPassManager &pm) {
  // Try to eliminate as many `tensorrt.broadcast` ops as possible.
  pm.addPass(tensorrt::createBroadcastEliminationPass());
  addCleanupPasses(pm);
  pm.addPass(tensorrt::createTransposeReshapeEliminationPass());
  addCleanupPasses(pm);
  pm.addPass(tensorrt::createRaiseNormalizationsPass());
  addCleanupPasses(pm);
}

void tensorrt::buildTensorRTModuleTransformationPipeline(
    OpPassManager &pm,
    const ApplyWorkaroundsPassOptions &bugWorkaroundOptions) {
  // Try to simplify the code and eliminate broadcast/transposes.
  buildTensorRTModuleSimplificationPipeline(pm);

  // Apply workarounds. This currently must be done before expanding ops because
  // WARs may use auxillary ops such as ExpandRank/CollapseRank ops.
  pm.addPass(tensorrt::createApplyWorkaroundsPass(bugWorkaroundOptions));
  addCleanupPasses(pm);

  // Run `tensorrt-expand-ops` twice with canonicalization in between. There
  // should not be any canonicalization between the second one and translation
  // to avoid the possibility that TRT extension operations are created from
  // canonicalizers, which is allowed. We need canonicalization after the first
  // call since ops like Shuffle have many useful canonicalizers.
  pm.addPass(tensorrt::createExpandOpsPass());
  addCleanupPasses(pm);
  pm.addPass(tensorrt::createExpandOpsPass());
  // Insert workarounds for int8 support.
  pm.addNestedPass<func::FuncOp>(tensorrt::createLegalizeInt8Pass());
}
