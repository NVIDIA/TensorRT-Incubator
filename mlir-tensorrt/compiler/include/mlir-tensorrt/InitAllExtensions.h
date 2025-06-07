//===- RegisterMlirTensorRtCoreDialects.h -----------------------*- C++ -*-===//
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
/// Registration methods for dialect extensions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_REGISTRATION_INITLLVMEXTENSIONS
#define MLIR_TENSORRT_REGISTRATION_INITLLVMEXTENSIONS

#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.h"
#include "mlir-tensorrt/Conversion/CUDAToLLVM/CUDAToLLVM.h"
#include "mlir-tensorrt/Conversion/PlanToLLVM/PlanToLLVM.h"
#include "mlir-tensorrt/Conversion/TensorRTRuntimeToLLVM/TensorRTRuntimeToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"

namespace mlirtrt::compiler {

inline void registerAllExtensions(mlir::DialectRegistry &registry) {
  // Register all conversion to LLVM interfaces.
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertCUDAToLLVMPatternInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertPlanToLLVMPatternInterface(registry);
  mlir::registerConvertTensorRTRuntimeToLLVMPatternInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);

  // Inliner extensions.
  mlir::func::registerInlinerExtension(registry);

  // Plan Extensions.
  mlir::plan::registerHostBackend(registry);
  mlir::plan::registerTensorRTBackend(registry);
}
} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_REGISTRATION_INITLLVMEXTENSIONS
