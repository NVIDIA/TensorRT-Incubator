//===- BufferizationOpInterfaceImpls.h -----------------------------------===//
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
#ifndef MLIR_EXECUTOR_TRANSFORMS_BUFFERIZATIONOPINTERFACEIMPLS
#define MLIR_EXECUTOR_TRANSFORMS_BUFFERIZATIONOPINTERFACEIMPLS

namespace llvm {
struct LogicalResult;
}

namespace mlir {
class DialectRegistry;
class FunctionOpInterface;

namespace bufferization {
struct BufferizationOptions;
}

namespace executor {

/// Register Bufferization-related op interface external models for Executor
/// dialect operations.
void registerBufferizationOpInterfaceExternalModels(DialectRegistry &registry);

/// Bufferize the ABI wrapper function type. This should be called on all
/// Executor ABI wrapper functions as a post-bufferization action.
llvm::LogicalResult bufferizeABIWrapperFunctionType(
    FunctionOpInterface abiFuncOp,
    const bufferization::BufferizationOptions &options);

} // namespace executor
} // namespace mlir

#endif // MLIR_EXECUTOR_TRANSFORMS_BUFFERIZATIONOPINTERFACEIMPLS
