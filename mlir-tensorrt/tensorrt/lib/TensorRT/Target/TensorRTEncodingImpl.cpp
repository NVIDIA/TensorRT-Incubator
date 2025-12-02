//===- TensorRTEncodingImpl.cpp -------------------------------------------===//
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
/// Implementation of the TensorRTEncodingOpInterface for all TensorRT dialect
/// operations. This code is mostly just generated from the Tablegen ops file
/// from the `trtLayerAdd` field in each op record.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.h"
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::tensorrt;

//===----------------------------------------------------------------------===//
// Helpers used in the tablegen'd code below.
//===----------------------------------------------------------------------===//

/// Create a for-loop style induction variable as a TensorRT loop recurrence
/// layer. This is a helper used in the ForOp translation in
/// `NetworkEncoder.inc.cpp`. This also sets the loop trim limit. Returns
/// the `ITensor*` that the `iv` is remapped to.
static nvinfer1::ITensor *
addForStyleInductionVariable(nvinfer1::INetworkDefinition *net,
                             nvinfer1::ILoop *loop, nvinfer1::ITensor *lb,
                             nvinfer1::ITensor *ub, nvinfer1::ITensor *step) {
  nvinfer1::IRecurrenceLayer *ivRec = loop->addRecurrence(*lb);
  nvinfer1::ITensor *ivBlockArg = ivRec->getOutput(0);
  // Create `iv + step -> iv`.
  nvinfer1::ITensor *ivYield =
      net->addElementWise(*ivBlockArg, *step,
                          ::nvinfer1::ElementWiseOperation::kSUM)
          ->getOutput(0);
  ivRec->setInput(1, *ivYield);
  // Create `break if iv >= ub`.
  nvinfer1::ITensor *condition =
      net->addElementWise(*ivBlockArg, *ub,
                          ::nvinfer1::ElementWiseOperation::kLESS)
          ->getOutput(0);
  loop->addTripLimit(*condition, ::nvinfer1::TripLimit::kWHILE);

  return ivBlockArg;
}

// Adds a RecurrenceLayer for each value in `blockArgs` with initial values
/// given by `initialValues` and uses `e` to remap `blockArgs`
static SmallVector<nvinfer1::ILayer *>
addRecurrenceLayersForBlockArgs(nvinfer1::ILoop *loop,
                                ArrayRef<nvinfer1::ITensor *> initialValues) {
  SmallVector<nvinfer1::ILayer *> recurrenceLayers;
  for (nvinfer1::ITensor *initOperand : initialValues)
    recurrenceLayers.push_back(loop->addRecurrence(*initOperand));
  return recurrenceLayers;
}

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

// Add IReccurrenceLayers for each loop carry variable in the while loop
// specification. The initial values in WhileOp are specified as operands to the
// WhileOp. IIdentityLayers wrap around the recurrence layers to prevent issues
// with recurrence layer interdependency.
static void addRecurrenceLayersForWhileLoopCarries(
    nvinfer1::INetworkDefinition *net, nvinfer1::ILoop *loop,
    ArrayRef<nvinfer1::ITensor *> whileOpOperands,
    SmallVector<nvinfer1::ILayer *> &recurrenceLayers,
    SmallVector<nvinfer1::ILayer *> &identityLayers) {

  for (nvinfer1::ITensor *initOperand : whileOpOperands) {
    auto layer = loop->addRecurrence(*initOperand);
    nvinfer1::IIdentityLayer *identityLayer =
        net->addIdentity(*layer->getOutput(0));
    if (!nvinfer1::adaptor::isStronglyTypedFlagEnabled(net))
      identityLayer->setOutputType(0, layer->getOutputType(0));
    recurrenceLayers.push_back(layer);
    identityLayers.push_back(identityLayer);
  }
}

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

//===----------------------------------------------------------------------===//
// Tablegen'd op interface definitions
//===----------------------------------------------------------------------===//

#define GEN_TRT_ENCODE_OP_IMPL

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-variable"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#endif
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.inc.cpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

void tensorrt::registerTensorRTEncodingOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, tensorrt::TensorRTDialect *dialect) {
#define GEN_TRT_ENCODE_OP_IMPL_ATTACH_INTERFACE
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.inc.cpp"
      });
}
