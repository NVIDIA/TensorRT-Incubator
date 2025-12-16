//===- InitialTransformSchedule.cpp ---------------------------------------===//
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
// Implementation of pass to generate transform dialect for the Linalg dialect
// converted from TensorRT.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-kernel/Kernel/IR/Configuration.h"
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/TransformSchedules/Passes.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformSchedules.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kernel-initial-transform-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")
#define DBGV(x, ...) LLVM_DEBUG(DBGS() << llvm::formatv(x "\n", __VA_ARGS__))

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_INITIALTRANSFORMSCHEDULEPASS
#include "mlir-kernel/Kernel/TransformSchedules/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace kernel;

namespace {
/// Encapsulates the root operation and the desired schedule generation
/// parameters for that operation.
struct DetermineTransformScheduleResult {
  DetermineTransformScheduleResult(
      TilingInterface rootOp, const TransformScheduleBase *scheduleGenerator,
      Attribute params)
      : rootOp(rootOp), scheduleGenerator(scheduleGenerator),
        parameters(params) {}

  TilingInterface rootOp;
  const TransformScheduleBase *scheduleGenerator;
  Attribute parameters;
};
} // namespace

/// Locate the root operation by tracing back from the function's terminator op.
/// The root linalg operation is the first linalg op on that path.
static FailureOr<TilingInterface> findRootOp(func::FuncOp funcOp) {
  Operation *term = funcOp.getBody().front().getTerminator();
  if (term->getNumOperands() != 1 &&
      !llvm::all_of(term->getOperands(), [&](Value operand) {
        return operand.getDefiningOp() ==
               term->getOperands().front().getDefiningOp();
      })) {
    return funcOp->emitOpError(
        "expected function return operands to be produced by a single op");
  }

  // Search backwards through reshape operations to identify root linalg
  // operation.
  Operation *rootCandidate = term->getOperand(0).getDefiningOp();
  while (rootCandidate) {
    if (auto tilingOp = dyn_cast<TilingInterface>(rootCandidate))
      return tilingOp;
    if (isa<tensor::ReshapeOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
            tensor::ExtractSliceOp>(rootCandidate)) {
      rootCandidate = rootCandidate->getOperand(0).getDefiningOp();
      continue;
    }

    // We may encounter `tensor.insert_slice` operations directly connected to
    // the return op. This can happen when, for example, a
    // `stablehlo.dynamic_update_slice` is pulled into a codegen cluster. For
    // now, we check the source/dest operands to see which one is not a block
    // argument. In general this may not resolve all cases but is currently
    // sufficient for ours.
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(rootCandidate)) {
      if (isa<BlockArgument>(insertSliceOp.getSource()) &&
          !isa<BlockArgument>(insertSliceOp.getDest())) {
        rootCandidate = insertSliceOp.getDest().getDefiningOp();
        continue;
      }
      if (isa<BlockArgument>(insertSliceOp.getDest()) &&
          !isa<BlockArgument>(insertSliceOp.getSource())) {
        rootCandidate = insertSliceOp.getSource().getDefiningOp();
        continue;
      }
    }
    return rootCandidate->emitOpError(
        "could not handle this operation when looking for root op");
  }
  return failure();
}

/// Decide the transform schedule parameters for different ops.
static FailureOr<DetermineTransformScheduleResult>
decideTransformScheduleParameters(const TransformScheduleSelector &selector,
                                  func::FuncOp funcOp, TilingInterface rootOp) {
  LLVM_DEBUG(DBGS() << "selected root:\n" << rootOp << "\n");

  FailureOr<const TransformScheduleBase *> generator =
      selector.getScheduleGenerator(rootOp);
  if (failed(generator))
    return (emitError(funcOp->getLoc())
            << "failed to match a code generation strategy for function '"
            << funcOp.getName() << "' when creating transform schedule")
               .attachNote(rootOp->getLoc())
           << "see root operation: " << rootOp;

  LLVM_DEBUG(DBGS() << "selected schedule: " << (*generator)->getMnemonic()
                    << "\n");

  FailureOr<Attribute> params =
      (*generator)->decideParameters(rootOp, selector.getOptions());
  if (failed(params))
    return rootOp->emitOpError("failed to determine schedule parameters");

  return DetermineTransformScheduleResult(rootOp, *generator, *params);
}

/// Generate transform schedule for the given function using the parameters
/// which were derived in the first step.
static LogicalResult
generateTransformSchedule(RewriterBase &b, func::FuncOp funcOp,
                          DetermineTransformScheduleResult params,
                          const TransformScheduleOptions &options) {
  OpBuilder::InsertionGuard g(b);
  Location loc = funcOp->getLoc();
  b.setInsertionPointAfter(funcOp);

  // Tag the root operation with the parameters attribute.
  if (params.parameters)
    params.rootOp->setAttr(kParametersAttrName, params.parameters);

  // Create the SequenceOp
  MLIRContext *ctx = b.getContext();
  auto seq = b.create<transform::SequenceOp>(
      loc, TypeRange(), transform::FailurePropagationMode::Propagate, nullptr,
      ValueRange());
  {
    // Specify which function to apply this transform schedule to
    seq->setAttr(b.getStringAttr(kTargetFuncAttrName),
                 SymbolRefAttr::get(funcOp));
    b.createBlock(&seq.getBody(), seq.getBody().begin(),
                  {transform::AnyOpType::get(ctx)}, {loc});
  }
  b.setInsertionPointToStart(seq.getBodyBlock());

  // Generate the schedule body.
  Value hostFuncHandle = seq.getBodyBlock()->getArgument(0);
  FailureOr<Value> kernelFuncHandle =
      params.scheduleGenerator->generateSchedule(
          b, loc, params.rootOp, hostFuncHandle, params.parameters, options);
  if (failed(kernelFuncHandle))
    return emitError(loc)
           << "failed to generate transform schedule for function '"
           << funcOp.getName() << "'";

  b.create<transform::YieldOp>(loc);
  return success();
}

/// Parses a list of "name:benefit" strings into a map.
static FailureOr<llvm::StringMap<int64_t>>
parseGeneratorBenefits(MLIRContext *ctx,
                       ArrayRef<std::string> generatorBenefits) {
  // clang-format off
  static llvm::SmallDenseSet<StringRef> validGenerators = {
      "fallback",
      "scatter",
  };
  // clang-format on

  llvm::StringMap<int64_t> benefits;
  for (const std::string &benefitStr : generatorBenefits) {
    auto split = StringRef(benefitStr).split(':');
    if (split.first.empty() || split.second.empty()) {
      return emitError(UnknownLoc::get(ctx))
             << "invalid generator benefit string: " << benefitStr << "\n";
    }
    if (validGenerators.find(split.first) == validGenerators.end()) {
      return emitError(UnknownLoc::get(ctx))
             << "invalid generator name: " << split.first.str() << "\n";
    }

    // Check for duplicate generator names
    if (benefits.find(split.first.str()) != benefits.end()) {
      return emitError(UnknownLoc::get(ctx))
             << "duplicate generator name: " << split.first.str() << "\n";
    }

    int64_t benefit;
    if (StringRef(split.second).getAsInteger(10, benefit)) {
      return emitError(UnknownLoc::get(ctx))
             << "invalid benefit integer: " << split.second << "\n";
    }
    benefits[split.first.str()] = benefit;
  }
  return benefits;
}

static LogicalResult registerTransformScheduleGeneratorWithBenefit(
    MLIRContext *ctx, std::shared_ptr<TransformScheduleRegistry> registry,
    StringRef name, int64_t benefit) {
  std::unique_ptr<TransformScheduleBase> generator = nullptr;
  if (name == "fallback") {
    generator = createFallbackTransformSchedule(ctx, benefit);
  } else if (name == "scatter") {
    generator = createScatterTransformSchedule(ctx, benefit);
  }

  if (!generator)
    return failure();
  registry->registerTransformScheduleGenerator(std::move(generator));
  return success();
}

/// Construct the registry of available transform schedules.
static std::shared_ptr<TransformScheduleRegistry>
constructTransformScheduleRegistry(
    MLIRContext *ctx, const llvm::StringMap<int64_t> &generatorBenefits) {

  auto registry = std::make_shared<TransformScheduleRegistry>();

  if (!generatorBenefits.empty()) {
    for (const auto &[name, benefit] : generatorBenefits) {
      if (failed(registerTransformScheduleGeneratorWithBenefit(
              ctx, registry, name, benefit))) {
        LLVM_DEBUG(DBGS() << "failed to register transform schedule generator: "
                          << name << "\n");
        continue;
      }
    }
    return registry;
  }

  // If no generator benefits are provided, use the default benefits.
  const int64_t baseBenefit = 2;
  registry->registerTransformScheduleGenerator(
      createFallbackTransformSchedule(ctx, baseBenefit));

  // Priority for Scatter schedule doesn't matter since its the only one that
  // can handle scatter for now.
  registry->registerTransformScheduleGenerator(
      createScatterTransformSchedule(ctx, baseBenefit));

  return registry;
}

//===----------------------------------------------------------------------===//
// InitialTransformSchedulePass
//===----------------------------------------------------------------------===//
namespace {
struct InitialTransformSchedulePass
    : public kernel::impl::InitialTransformSchedulePassBase<
          InitialTransformSchedulePass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *ctx) override {
    auto benefits = parseGeneratorBenefits(ctx, generatorBenefit);
    if (failed(benefits)) {
      return emitError(UnknownLoc::get(ctx))
             << "failed to parse generator benefits, expected format: "
                "'name:benefit'\n";
    }
    registry = constructTransformScheduleRegistry(ctx, *benefits);
    return success();
  }

  void runOnOperation() override {

    int64_t computeCapability = this->computeCapability;
    int64_t maxSharedMemoryPerBlockKb = this->maxSharedMemoryPerBlockKb;
    if (inferDeviceInfoFromHost) {
      mtrt::StatusOr<mtrt::DeviceInfo> devInfo =
          mtrt::getDeviceInformationFromHost(/*cudaDeviceOrdinal=*/0);
      if (!devInfo.isOk()) {
        emitError(UnknownLoc::get(&getContext()))
            << "failed to determine host GPU information: "
            << devInfo.getStatus().getMessage();
        return signalPassFailure();
      }
      computeCapability = devInfo->computeCapability;
      maxRegistersPerBlock = devInfo->maxRegistersPerBlock;
      maxSharedMemoryPerBlockKb = devInfo->maxSharedMemoryPerBlockKb;

      DBGV("inferred device info[cc={0}, regsPerBlock={1}, "
           "maxSharedMemPerBlockKb={2}",
           computeCapability, maxRegistersPerBlock, maxSharedMemoryPerBlockKb);
    }

    ModuleOp module = getOperation();

    // TODO: this data layout should conform to the data layout for the *kernel*
    // module, but that doesn't yet exist (it is created when applying the
    // transform IR generated by this pass). We could directly create the data
    // layout for the GPU target based on the device information, but for most
    // purposes the default data layout works fine.
    DataLayoutAnalysis &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const mlir::DataLayout &dataLayout =
        dataLayoutAnalysis.getAtOrAbove(module);

    /// Declare a TransformScheduleSelector, which assists with picking
    auto target = NVVM::NVVMTargetAttr::get(
        &getContext(), /*optLevel=*/2, /*triple=*/"nvptx64-nvidia-cuda",
        /*arch=*/"sm_" + std::to_string(computeCapability));

    TransformScheduleSelector selector(
        *registry, TransformScheduleOptions{
                       static_cast<uint64_t>(maxSharedMemoryPerBlockKb) * 1024,
                       0, maxRegistersPerBlock, dataLayout,
                       cast<gpu::TargetAttrInterface>(target)});

    // Iter through all the functions in the module and generate transform
    // schedules for each if it contains a "linalg op".
    IRRewriter rewriter(module->getContext());
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration() ||
          (!funcFilter.empty() && !funcOp->getAttr(funcFilter)))
        continue;

      // Enumerate the linalg operations.
      if (funcOp.getBody().front().getOps<TilingInterface>().empty())
        continue;

      // Find and mark the root operation.
      FailureOr<TilingInterface> rootOp = findRootOp(funcOp);
      if (failed(rootOp))
        continue;
      (*rootOp)->setAttr(kRootGenericAttrName,
                         UnitAttr::get(rootOp->getContext()));

      FailureOr<DetermineTransformScheduleResult> scheduleTypeResult =
          decideTransformScheduleParameters(selector, funcOp, *rootOp);
      if (failed(scheduleTypeResult))
        return signalPassFailure();

      // Generate the transform IR for this function.
      if (failed(generateTransformSchedule(rewriter, funcOp,
                                           std::move(*scheduleTypeResult),
                                           selector.getOptions())))
        return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);

    // Register the builtin transform schedule generators. These are simply
    // callbacks which are invoked prior to this pass running (if they have not
    // already been loaded).
    // Custom schedule generators must be registered prior to invoking the pass.
    mlir::kernel::registerBuiltinTransformScheduleGenerators(registry);
  }

  /// The registry of all transform schedule generators.
  std::shared_ptr<TransformScheduleRegistry> registry;
};
} // namespace
