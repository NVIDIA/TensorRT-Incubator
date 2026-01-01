//===- KernelBackend.cpp --------------------------------------------------===//
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
/// Implementation of the Kernel backend.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mtrt::compiler;
using namespace mlir;
using namespace mlir::plan;

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Backends/Kernel/KernelBackendAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// KernelBackendAttr
//===----------------------------------------------------------------------===//

/// ClusteringOpts that identifies groups of codegen-able operations and
/// will be clustered into one Codegen cluster.
static ClusteringOpts
getCodegenClusteringOpts(MLIRContext *ctx, DataFlowSolver &solver,
                         plan::KernelBackendAttr clusterKind) {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return false; };
  opts.clusterTarget = clusterKind;
  opts.bfsRootTraversal = ClusteringRootTraversalDirection::PostOrder;
  opts.shouldGrowClusterFn =
      [](Operation *producer, ClusterRange producerCluster, Operation *consumer,
         ClusterRange consumerCluster) { return true; };
  opts.isClusterableOp = [&solver](Operation *op) {
    // Exclude constants since these are cloned into regions.
    if (op->hasTrait<OpTrait::ConstantLike>())
      return false;

    if (!isa<stablehlo::StablehloDialect>(op->getDialect()))
      return false;

    // Don't cluster oprerations nested within e.g. stablehlo.reduce bodies.
    if (isa<stablehlo::StablehloDialect>(op->getParentOp()->getDialect()))
      return false;

    if (stablehlo::canConvertToLinalg(op))
      return !mtrt::compiler::detail::shouldRunOnHost(op, solver);

    if (isa<stablehlo::ScatterOp, kernel::ScatterOp>(op))
      return true;

    return false;
  };
  return opts;
}

FailureOr<ClusteringOpts> plan::KernelBackendAttr::getClusterKindOptions(
    InputKind inputKind, Operation *op, DataFlowSolver &solver) const {
  return getCodegenClusteringOpts(getContext(), solver, *this);
}

/// Return a 1-to-N type converter for scalarizing Tensor types to unpacked
/// scalar types.
static TypeConverter getSignedToSignlessConverter() {
  // Add a type converter, target and source materialization to convert
  // `tensor<1xdtype>` to `dtype` and back.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type t) -> std::optional<Type> { return t; });
  typeConverter.addConversion(
      [&](Type t,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        auto rtt = dyn_cast<RankedTensorType>(t);
        if (!rtt || (!rtt.getElementType().isSignedInteger() &&
                     !rtt.getElementType().isUnsignedInteger()))
          return std::nullopt;
        result.push_back(RankedTensorType::get(
            rtt.getShape(),
            IntegerType::get(t.getContext(),
                             rtt.getElementType().getIntOrFloatBitWidth()),
            rtt.getEncoding()));
        return success();
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
         Location loc) -> SmallVector<Value> {
        if (inputs.size() != 1)
          return {};
        Value input = inputs.front();
        auto rtt = dyn_cast<RankedTensorType>(input.getType());
        if (!rtt || (!rtt.getElementType().isSignedInteger() &&
                     !rtt.getElementType().isUnsignedInteger()))
          return {};
        SmallVector<Value> result;
        llvm::append_range(
            result,
            builder.create<UnrealizedConversionCastOp>(loc, resultTypes, input)
                .getResults());
        return result;
      });
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs, Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  return typeConverter;
}

std::optional<OutlineRegionOptions>
plan::KernelBackendAttr::getClusterOutliningOptions(
    InputKind inputKind, MLIRContext *ctx, SymbolTable &symbolTable) const {
  OpBuilder b(ctx);
  return OutlineRegionOptions{
      /*typeConverter=*/getSignedToSignlessConverter(),
      /*shouldCloneProducer=*/
      [this](Value v, Region &targetRegion) -> bool {
        return this->shouldCloneProducer(targetRegion.getParentOp(),
                                         v.getDefiningOp());
      },
      /*createFunc=*/
      OutlineRegionOptions::getDefaultCreateFuncAndCallStubFunc(
          symbolTable, /*extraFuncAttrs=*/{}, "codegen_cluster")};
}

int64_t plan::KernelBackendAttr::getClusterBenefit(InputKind inputKind) const {
  return getBenefit();
}

std::function<bool(const Cluster &)>
plan::KernelBackendAttr::getClusterFilter(InputKind inputKind) const {
  return [](const Cluster &cluster) { return true; };
}

bool plan::KernelBackendAttr::supportsInputKind(InputKind inputKind) const {
  return inputKind == InputKind::Stablehlo;
}

bool plan::KernelBackendAttr::requiresInputBoundsForDynamicShapes(
    bool useDestinationStyleCallingConvention) const {
  // KernelBackend does not require shape bounds for dynamically shaped input
  // tensors.
  return false;
}

bool plan::KernelBackendAttr::requiresOutputBoundsForDynamicShapes(
    bool useDestinationStyleCallingConvention) const {
  // KernelBackend does not require shape bounds for dynamically shaped output
  // tensors.
  return false;
}

//===----------------------------------------------------------------------===//
// Extension Registration
//===----------------------------------------------------------------------===//

namespace {
class PlanDialectCodegenExtension
    : public plan::PlanDialectExtension<PlanDialectCodegenExtension> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanDialectCodegenExtension)

  void init() {
    (void)&generatedAttributeParser;
    (void)&generatedAttributePrinter;
    registerAttributes<plan::KernelBackendAttr>();

    // clang-format off
    declareGeneratedDialects<
      mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,
      mlir::linalg::LinalgDialect,
      mlir::tensor::TensorDialect,
      mlir::func::FuncDialect,
      mlir::math::MathDialect,
      mlir::scf::SCFDialect>();
    // clang-format on
  }
};
} // namespace

void mtrt::compiler::registerKernelBackend(DialectRegistry &registry) {
  registry.addExtensions<PlanDialectCodegenExtension>();
}

llvm::StringRef mtrt::compiler::getKernelGenClusterAttrName() {
  return "cluster.codegen";
}
