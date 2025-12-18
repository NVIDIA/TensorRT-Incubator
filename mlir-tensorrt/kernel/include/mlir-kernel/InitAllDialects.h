//===- InitAllDialects.h --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Registration functions for registering dialects and interface
/// implementations only suitable for generating GPU kernels from the Kernel
/// dialect. Registrations here do not include dialects or interfaces required
/// for the TensorRT, Plan, or Executor dialects.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_INITALLDIALECTS
#define MLIR_KERNEL_INITALLDIALECTS

#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.h"
#include "mlir-kernel/Kernel/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-kernel/Kernel/Transforms/KernelToLLVMIRTranslation.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace mlir::kernel {

inline void registerAllRequiredDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::complex::ComplexDialect,
                  mlir::DLTIDialect,
                  mlir::emitc::EmitCDialect,
                  mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect,
                  mlir::index::IndexDialect,
                  mlir::kernel::KernelDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::nvgpu::NVGPUDialect,
                  mlir::NVVM::NVVMDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::transform::TransformDialect,
                  mlir::ub::UBDialect,
                  mlir::vector::VectorDialect
                  >();
  // clang-format on

  // Register all external models.
  // Register pointer-like type interfaces for external pointer types.
  mlir::kernel::registerExternalDialectPtrTypeInterfaceImpls(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::builtin::registerCastOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::kernel::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::kernel::registerGPUModuleLoweringAttrExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerFindPayloadReplacementOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::linalg::registerSubsetOpInterfaceExternalModels(registry);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);

  // Register misc dialect extensions.
  func::registerInlinerExtension(registry);

  // Register translation dialect interfaces
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Register all transform dialect extensions.
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::kernel::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  mlir::transform::registerLoopExtension(registry);

  // Register ConvertToLLVM dialect extensions..
  cf::registerConvertControlFlowToLLVMInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  NVVM::registerConvertGpuToNVVMInterface(registry);
}
} // namespace mlir::kernel

#endif // MLIR_KERNEL_INITALLDIALECTS
