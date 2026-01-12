//===- EinsumHelper.h -------------------------------------------*- c++ -*-===//
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
// Helper functions for einsum operation
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_EINSUMHELPER
#define MLIR_TENSORRT_UTILS_EINSUMHELPER

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tensorrt {
namespace einsum {

using ErrorFn = function_ref<FailureOr<InFlightDiagnostic>(
    std::optional<Location> loc, const llvm::Twine &message)>;
using Subscript = llvm::SmallString<4>;
using InputSubscripts = llvm::SmallVector<Subscript, 2>;
struct IOSubscripts {
  InputSubscripts inputs;
  Subscript outputs;
};

/// Verify input/output subscripts and associated types. This function calls
/// all helper functions mentioned above for this purpose and is called within
/// `tensorrt::EinsumOp::verify()`.
LogicalResult verify(StringRef equation, TypeRange inputOperands,
                     TensorType output, std::optional<Location> loc,
                     ErrorFn emitErrorFn);

/// Infers output shape given user passed einsum equation and input operands.
/// It validates equation, parse subscripts, validates subscripts and compute
/// output subscript if not present in the equation, all with the help of
/// functions mentioned above. Once validated output subscript is available,
/// output shape is inferred by looking into input <label, dimension> map.
FailureOr<SmallVector<int64_t>> inferOutputShape(StringRef equation,
                                                 TypeRange inputOperands,
                                                 std::optional<Location> loc,
                                                 ErrorFn emitErrorFn);

} // namespace einsum
} // namespace tensorrt
} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_EINSUMHELPER
