//===- HostBackend.cpp ----------------------------------------------------===//
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
/// Definitions for the Host backend Plan dialect extension.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::plan;

#define DEBUG_TYPE "host-backend"
#define DBGS() llvm::dbgs() << "[" << DEBUG_TYPE << "] "

/// Include the Tablegen'd C++ code describing the backend attribute. We will
/// attach this to the plan dialect as an extension.
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Backends/Host/HostBackendAttrs.cpp.inc"

static bool isScalarizableType(Type t) {
  auto rtt = dyn_cast<RankedTensorType>(t);
  if (!rtt || !rtt.hasStaticShape() ||
      !rtt.getElementType().isIntOrIndexOrFloat())
    return false;
  return rtt.getNumElements() <= 16;
}

/// Returns true if the given operation should run "on the host". This means
/// that the operation can be converted to Executor IR. It derives this
/// information based on the operation, the operands, and the TensorKindAnalysis
/// information.
bool plan::detail::shouldRunOnHost(Operation *op,
                                   const DataFlowSolver &solver) {
  // An operation can't be placed on the host if the types are too big.
  LLVM_DEBUG(DBGS() << "should run on host? " << *op << "\n");
  if (!llvm::all_of(op->getResultTypes(), isScalarizableType) ||
      !llvm::all_of(op->getOperandTypes(), isScalarizableType)) {
    LLVM_DEBUG(DBGS() << "  types not all host compatible\n");
    return false;
  }

  /// TODO: remove unconditional dependence on StableHlo dialect.

  // Filter for StableHLO dialect ops. Don't consider stablehlo ops nested in
  // other stablehlo ops.
  if (!isa<stablehlo::StablehloDialect>(op->getDialect()) ||
      isa<stablehlo::StablehloDialect>(op->getParentOp()->getDialect())) {
    LLVM_DEBUG(DBGS() << "  not stablehlo op\n");
    return false;
  }

  // Ignore constants. We don't cluster constants. They are cloned during the
  // outlining step.
  if (op->hasTrait<OpTrait::ConstantLike>())
    return false;

  // Filter for which operations we support on the host.
  if (isa_and_present<stablehlo::StablehloDialect>(op->getDialect())) {
    if (!stablehlo::canConvertToLinalg(op))
      return false;
  }

  // If the operation doesn't have any operands, then we can run on host if
  // the result is required on host (e.g. `stablehlo.arange : tensor<4xi32>`).
  if (op->getNumOperands() == 0) {
    LLVM_DEBUG(DBGS() << "  checking result TensorKinds\n");
    return llvm::all_of(op->getResults(), [&](Value v) {
      const auto *lattice = solver.lookupState<TensorKindLattice>(v);
      LLVM_DEBUG({
        DBGS() << "  arg: ";
        if (lattice)
          lattice->print(llvm::dbgs());
        else
          llvm::dbgs() << "<nullptr>";
        llvm::dbgs() << "\n";
      });
      return lattice && !lattice->getValue().isUninitialized() &&
             lattice->getValue().isHostVisible();
    });
  }

  // If all the types are small enough and they are host tensors, then we can
  // place the computation on the host. Note that the TensorKind of the
  // results doesn't matter here. If the operands and result types are small,
  // then we can run the computation on the host as long as the inputs are on
  // the host. A result TensorKind of 'device' or 'both' just means the result
  // must be transferred to the device afterwards.
  LLVM_DEBUG(DBGS() << "  checking operand TensorKinds\n");
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(operand);
    LLVM_DEBUG({
      DBGS() << "  arg: ";
      if (lattice)
        lattice->print(llvm::dbgs());
      else
        llvm::dbgs() << "<nullptr>";
      llvm::dbgs() << "\n";
    });
    return lattice && !lattice->getValue().isUninitialized() &&
           lattice->getValue().isHostVisible();
  });
}

//===----------------------------------------------------------------------===//
// HostBackendAttr
//===----------------------------------------------------------------------===//

int64_t HostBackendAttr::getClusterBenefit(InputKind inputKind) const {
  return getBenefit();
}

/// Return true if the op is likely in a "compute" region, like the region of
/// `stablehlo.reduce` or `linalg.generic`. For the purposes of this pass, we're
/// defining "compute" region as a region where the normal flow-of-control does
/// not enter from outside. It's only used to define the semantics of the parent
/// operation.a
static bool inComputeRegion(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    // If the parent is a function, then we're in a normal region.
    if (isa<FunctionOpInterface>(parent))
      return false;
    // We are in a region which is not a control flow region and not a function,
    // so it's probably a "compute" region.
    if (!isa<RegionBranchOpInterface>(parent))
      return true;
    // If we're in a control flow region, we may still be nested in a "compute"
    // region. E.g. `scf.if` is allowed in `linalg.generic` region. Keep going
    // up to find the parent function.
    parent = parent->getParentOp();
  }
  return false;
}

/// ClusteringOpts that identifies groups of `stablehlo` ops that can be
/// converted to scalars and will be clustered into scalar cluster.
FailureOr<ClusteringOpts>
HostBackendAttr::getClusterKindOptions(InputKind inputKind, Operation *op,
                                       DataFlowSolver &solver) const {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [](Operation *op) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      return false;

    if (inComputeRegion(op))
      return false;

    if (llvm::isa<plan::WithValuesOp>(op))
      return true;
    if (llvm::isa_and_present<stablehlo::StablehloDialect>(op->getDialect())) {
      return stablehlo::canConvertToLinalg(op);
    }
    if (llvm::isa_and_present<arith::ArithDialect, math::MathDialect>(
            op->getDialect()) &&
        llvm::all_of(op->getOperandTypes(),
                     llvm::IsaPred<IntegerType, FloatType>)) {
      return true;
    }
    if (llvm::isa<tensor::ExtractOp, tensor::InsertOp>(op)) {
      return true;
    }
    if (llvm::isa<ToLoopsOpInterface>(op))
      return true;
    return false;
  };
  return opts;
}

/// Determines whether a cluster being outlined should clone a constant or
/// pass constant by value.
static bool shouldCloneProducer(Value v, Region &cluster) {
  Operation *producer = v.getDefiningOp();
  if (!producer->hasTrait<OpTrait::ConstantLike>() ||
      producer->getNumResults() != 1)
    return false;
  RankedTensorType type =
      dyn_cast<RankedTensorType>(producer->getResultTypes().front());
  if (!type || !type.hasStaticShape())
    return false;

  // A value should be cloned if all of its uses are in the cluster.
  if (llvm::all_of(v.getUsers(), [&](Operation *user) {
        return cluster.isAncestor(user->getParentRegion());
      }))
    return true;
  return type.getNumElements() *
             llvm::divideCeil(type.getElementTypeBitWidth(), 8) <
         1024 * 1024;
}

/// Host regions do not require closre since we have no need for shape or value
/// bounds information.
bool HostBackendAttr::requiresClosure(InputKind) const { return false; }

std::optional<OutlineRegionOptions> HostBackendAttr::getClusterOutliningOptions(
    InputKind inputKind, MLIRContext *ctx,
    SymbolTable &moduleSymbolTable) const {
  OpBuilder b(ctx);
  return OutlineRegionOptions{
      /*typeConverter=*/getIdentityTypeConverter(),
      /*shouldCloneProducer=*/shouldCloneProducer,
      /*createFunc=*/
      OutlineRegionOptions::getDefaultCreateFuncAndCallStubFunc(
          moduleSymbolTable, /*extraFuncAttrs=*/{}, "host_backend")};
}

std::function<bool(const Cluster &)>
HostBackendAttr::getClusterFilter(InputKind) const {
  return [](const Cluster &cluster) {
    return !llvm::all_of(cluster, [](Operation *op) {
      return op->hasTrait<OpTrait::ConstantLike>() ||
             llvm::isa<plan::WithValuesOp>(op);
    });
  };
}

bool HostBackendAttr::supportsInputKind(InputKind inputKind) const {
  return inputKind == InputKind::Stablehlo;
}

MemorySpace HostBackendAttr::getDefaultMemorySpace() const {
  return MemorySpace::host;
}

//===----------------------------------------------------------------------===//
// Extension Registration
//===----------------------------------------------------------------------===//

namespace {
class PlanDialectHostBackend
    : public plan::PlanDialectExtension<PlanDialectHostBackend> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanDialectHostBackend)

  void init() {
    (void)&generatedAttributeParser;
    (void)&generatedAttributePrinter;
    registerAttributes<plan::HostBackendAttr>();

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

void mlir::plan::registerHostBackend(DialectRegistry &registry) {
  registry.addExtensions<PlanDialectHostBackend>();
}
