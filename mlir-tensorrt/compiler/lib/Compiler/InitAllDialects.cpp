//===- InitAllDialects.cpp ------------------------------------------------===//
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
/// Registration methods for MLIR dialects.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/InitAllDialects.h"

#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/BufferizationOpInterfaceImpls.h"
#include "mlir-tensorrt-common/Dialect/EmitCExt/IR/DataLayoutImpl.h"
#include "mlir-tensorrt-common/Dialect/LinalgExt/Transforms/ToLoopsOpInterfaceImpl.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-tensorrt/Conversion/CUDAToLLVM/CUDAToLLVM.h"
#include "mlir-tensorrt/Conversion/PlanToLLVM/PlanToLLVM.h"
#include "mlir-tensorrt/Conversion/TensorRTRuntimeToLLVM/TensorRTRuntimeToLLVM.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/StablehloExt/IR/StableHloExt.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-tensorrt/Features.h"
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Target/LLVM/NVVM/Target.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#endif

void mtrt::compiler::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
    registry.insert<
        mlir::affine::AffineDialect,
        mlir::arith::ArithDialect,
        mlir::async::AsyncDialect,
        mlir::bufferization::BufferizationDialect,
        mlir::cf::ControlFlowDialect,
        mlir::complex::ComplexDialect,
        mlir::cuda::CUDADialect,
        mlir::DLTIDialect,
        mlir::emitc::EmitCDialect,
        mlir::executor::ExecutorDialect,
        mlir::func::FuncDialect,
        mlir::gpu::GPUDialect,
        mlir::index::IndexDialect,
        mlir::linalg::LinalgDialect,
        mlir::LLVM::LLVMDialect,
        mlir::math::MathDialect,
        mlir::memref::MemRefDialect,
        mlir::NVVM::NVVMDialect,
        mlir::pdl_interp::PDLInterpDialect,
        mlir::pdl::PDLDialect,
        mlir::plan::PlanDialect,
        mlir::ptr::PtrDialect,
        mlir::quant::QuantDialect,
        mlir::scf::SCFDialect,
        mlir::shape::ShapeDialect,
        mlir::tensor::TensorDialect,
        mlir::tensorrt::TensorRTDialect,
        mlir::transform::TransformDialect,
        mlir::trtrt::TensorRTRuntimeDialect,
        mlir::ub::UBDialect,
        mlir::vector::VectorDialect
      >();
  // clang-format on

  IF_MLIR_TRT_ENABLE_HLO({
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
  });

  // Register all external models.
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::builtin::registerCastOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cuda::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::emitc_ext::registerDataLayoutInterfaceExternalModels(registry);
  mlir::executor::registerBufferizationOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::linalg::registerSubsetOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg_ext::registerToLoopsOpInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
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
  mlir::tensorrt::registerTensorKindOpInterfaceExternalModels(registry);
  mlir::trtrt::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);

  IF_MLIR_TRT_TARGET_TENSORRT({
    mlir::tensorrt::registerTensorRTEncodingOpInterfaceExternalModels(registry);
  });

  IF_MLIR_TRT_ENABLE_HLO({
    mlir::stablehlo::registerInferTensorValueRangeInterfaceExternalModels(
        registry);
    mlir::stablehlo::registerTensorKindOpInterfaceExternalModels(registry);
    mlir::stablehlo::registerTypeInferenceExternalModels(registry);
  });

  // Register CompilerTasks.
  static llvm::once_flag once;
  llvm::call_once(once, []() {
    mtrt::compiler::registerStableHloToExecutableTask();
    mtrt::compiler::registerTensorRTToExecutableTask();
  });
}

void mtrt::compiler::registerAllExtensions(mlir::DialectRegistry &registry) {
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

  // Compiler task extensions.
  mtrt::compiler::registerTensorRTExtension(registry);
}
