//===- PlanInterfaces.cpp -- ----------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Definitions for Plan op/attribute/type interfaces.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "plan-interfaces"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::plan;

/// Returns true if the given operation should run "on the host". This means
/// that the operation can be converted to Executor IR. It derives this
/// information based on the operation, the operands, and the TensorKindAnalysis
/// information.
bool plan::detail::shouldRunOnHost(Operation *op, DataFlowSolver &solver) {
  // An operation can't be placed on the host if the types are too big.
  LLVM_DEBUG(DBGS() << "should run on host? " << *op << "\n");
  auto isHostType = [](Type t) {
    return t.isIntOrIndexOrFloat() || stablehlo_ext::isScalarizableType(t);
  };
  if (!llvm::all_of(op->getResultTypes(), isHostType) ||
      !llvm::all_of(op->getOperandTypes(), isHostType)) {
    LLVM_DEBUG(DBGS() << "  types not all host compatible\n");
    return false;
  }

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
  if (!op->hasTrait<OpTrait::Elementwise>() &&
      !isa<stablehlo::ConcatenateOp, stablehlo::IotaOp, stablehlo::ReshapeOp,
           stablehlo::BroadcastInDimOp, stablehlo::SliceOp,
           stablehlo::BitcastConvertOp, stablehlo::ConvertOp,
           stablehlo::SelectOp, stablehlo::ReduceOp>(op)) {
    LLVM_DEBUG(DBGS() << "  not a supported op\n");
    return false;
  }

  // If the operation doesn't have any operands, then we can run on host if
  // the result is required on host (e.g. `stablehlo.arange : tensor<4xi32>`).
  if (op->getNumOperands() == 0) {
    LLVM_DEBUG(DBGS() << "  checking result TensorKinds\n");
    return llvm::all_of(op->getResults(), [&](Value v) {
      const auto *lattice = solver.lookupState<TensorKindLattice>(v);
      LLVM_DEBUG({
        if (lattice)
          DBGS() << "  arg: ";
        lattice->print(llvm::dbgs());
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
      if (lattice)
        DBGS() << "  arg: ";
      lattice->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    return lattice && !lattice->getValue().isUninitialized() &&
           lattice->getValue().isHostVisible();
  });
}
