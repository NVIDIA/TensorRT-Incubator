//===- PlanToLLVM.h ---------------------------------------------*- C++ -*-===//
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
///
/// Declarations for 'cuda-to-llvm' conversions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_CUDATOLLVM_CUDATOLLVM
#define MLIR_TENSORRT_CONVERSION_CUDATOLLVM_CUDATOLLVM

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class IRRewriter;
class ModuleOp;
class SymbolTableCollection;

/// Register the ConvertToLLVMPatternsInterface for the CUDA dialect.
void registerConvertCUDAToLLVMPatternInterface(DialectRegistry &registry);

/// Populate type conversions for CUDA dialect types to LLVM types.
void populateCUDAToLLVMTypeConversions(LLVMTypeConverter &typeConverter);

/// Populate op conversion patterns for CUDA dialect ops to LLVM ops.
void populateCUDAToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns);

LogicalResult lowerCUDAGlobalsToLLVM(IRRewriter &rewriter, ModuleOp rootOp,
                                     SymbolTableCollection &symbolTables);

} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_CUDATOLLVM_CUDATOLLVM
