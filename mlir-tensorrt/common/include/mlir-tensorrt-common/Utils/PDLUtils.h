//===- PDLUtils.h -----------------------------------------------*- C++ -*-===//
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
/// Utilities for the PDL dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_PDLUTILS
#define MLIR_TENSORRT_COMMON_UTILS_PDLUTILS

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

/// This utiliity is required in order to work around an upstream bug where
/// native PDLL `Constraint` and `Rewrite` functions declared as
/// `Constraint(op: Op<my_dialect.op>)` will produce an error in C++ compilation
/// since the `pdl_function_builder::ProcessPDLValue` adaptor for derived
/// operations is currently broken.
///
/// For each operation type that one would like to use in the argument of a
/// native PDLL `Constraint` or `Rewrite`, one must call
/// `DECL_PDL_ARG_ADAPTOR(OpClass)` inside the top-level `mlir` namespace before
/// including the generated PDLL C++ file.
///
/// TODO: remove this when the adaptor is fixed upstream.
template <typename OpType>
struct ProcessPDLValueForOpType {
  static LogicalResult
  verifyAsArg(function_ref<LogicalResult(const Twine &)> errorFn,
              PDLValue pdlValue, size_t argIdx) {
    if (pdlValue)
      return success();
    return errorFn("expected a non-null value for argument " + Twine(argIdx) +
                   " of type: " + llvm::getTypeName<OpType>());
  }

  static OpType processAsArg(PDLValue pdlValue) {
    return cast<OpType>(pdlValue.cast<Operation *>());
  }
  static void processAsResult(PatternRewriter &, PDLResultList &results,
                              OpType value) {
    results.push_back(value.getOperation());
  }
};

#define DECL_PDL_ARG_ADAPTOR(x)                                                \
  template <>                                                                  \
  struct detail::pdl_function_builder::ProcessPDLValue<x>                      \
      : public ProcessPDLValueForOpType<x> {};

} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_UTILS_PDLUTILS
