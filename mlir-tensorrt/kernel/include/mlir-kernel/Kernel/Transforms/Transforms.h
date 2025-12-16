//===- Transforms.h -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
///  Declarations for Kernel dialect transform functions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_TRANSFORMS_TRANSFORMS
#define MLIR_KERNEL_KERNEL_TRANSFORMS_TRANSFORMS

#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir {
class TypeConverter;
class ConversionTarget;
class RewritePatternSet;
class RewriterBase;

namespace vector {
class TransferReadOp;
}

namespace linalg {
class LinalgOp;
}

namespace NVVM {
class NVVMTargetAttr;
}

namespace kernel {

/// Register the bufferization scope interface implementation.
void registerKernelBufferizationScopeOpInterfaceImpls(
    DialectRegistry &registry);

/// Run bufferization on a `gpu.module` operation.
LogicalResult bufferizeKernelModule(
    gpu::GPUModuleOp op,
    bufferization::LayoutMapOption functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::FullyDynamicLayoutMap);

/// Populate patterns for rewrites of linalg operations that should occur before
/// schedule generation.
void populatePrepareLinalgForCodegenPatterns(RewritePatternSet &patterns);

/// Set the target attribute for a `gpu.module` operation.
LogicalResult setGPUTargets(gpu::GPUModuleOp op, ArrayRef<Attribute> targets);

/// Populate patterns for simplifying scalar arithmetic and affine operations
/// baed on bound analysis.
void populateAffineBoundsOptimizationPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Vector dialect-related Patterns
//===----------------------------------------------------------------------===//

/// This function detects when a `vector.transfer_read` is reading from the
/// result of a `vector.transfer_write` which has an overwrite/fill-like effect.
/// If so, the `vector.transfer_reads` is replaced with a splat `arith.constant`
/// of the corresponding vector type.
LogicalResult
replaceVectorTransferReadWithConstant(RewriterBase &rewriter,
                                      vector::TransferReadOp transferReadOp);

//===----------------------------------------------------------------------===//
// Bufferization Related Utilities
//===----------------------------------------------------------------------===//

/// Retrieve the base one-shot-bufferization options that should be used for a
/// 'gpu.module' operation.
bufferization::OneShotBufferizationOptions
getKernelModuleBufferizationOptions();

/// Run post-bufferization options. This includes things like insertion of
/// alignment hints, performing validation, and performing simple optimizations
/// that don't require extensive analysis.
LogicalResult
runKernelModulePostBufferizationActions(gpu::GPUModuleOp op,
                                        SymbolUserMap &symbolUserMap);

} // namespace kernel
} // namespace mlir

#endif // MLIR_KERNEL_KERNEL_TRANSFORMS_TRANSFORMS
