//===- RegisterMlirTensorRtDialects.h ---------------------------*- C++ -*-===//
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
// Register all dialects required by parts of this project, including dialects
// required by transformations or that are accepted by inputs.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTDIALECTS_H
#define MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTDIALECTS_H

#include "mlir-tensorrt/Registration/RegisterMlirTensorRtCoreDialects.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Dialect/StableHloExt/IR/StableHloExt.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#endif

#ifdef MLIR_TRT_ENABLE_EXECUTOR
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#endif

namespace mlir {

inline void registerAllMlirTensorRtExecutorDialects(DialectRegistry &registry) {
  // Registration for executor dialect and all upstream dialects that can appear
  // in the host IR.
  registry.insert<affine::AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                  bufferization::BufferizationDialect, math::MathDialect>();
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
}

inline void registerAllMlirTensorRtDialects(DialectRegistry &registry) {
  registerCoreMlirTensorRtDialects(registry);
  registerMlirTensorRtBufferizationInterfaces(registry);
  registerMlirTensorRtTransformExtensions(registry);

  // Register other dialects declared in upstream or in dependencies. Only
  // register dialects if absolutely necessary (i.e. they appear in the input
  // IR).
  registry.insert<arith::ArithDialect, pdl::PDLDialect, shape::ShapeDialect,
                  tensor::TensorDialect, mlir::quant::QuantDialect,
                  scf::SCFDialect, transform::TransformDialect>();

#ifdef MLIR_TRT_ENABLE_HLO
  registry.insert<mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
                  scf::SCFDialect, vhlo::VhloDialect>();
  stablehlo::registerTensorKindOpInterfaceExternalModels(registry);
  stablehlo::registerTypeInferenceExternalModels(registry);
#endif // MLIR_TRT_ENABLE_HLO

#ifdef MLIR_TRT_ENABLE_EXECUTOR
  registerAllMlirTensorRtExecutorDialects(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
#endif // MLIR_TRT_ENABLE_EXECUTOR
}

} // namespace mlir

#endif // MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTDIALECTS_H
