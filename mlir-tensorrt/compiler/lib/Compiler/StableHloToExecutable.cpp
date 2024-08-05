//===- StableHloToExecutable.cpp ------------------------------------------===//
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
/// MLIR-TensorRT CompilerClient API definitions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/StableHloToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/TensorRTExtension/TensorRTExtension.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Pipelines/StableHloInputPipelines.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "compiler-api"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#ifdef MLIR_TRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

static mlir::LogicalResult setupPassManager(mlir::PassManager &pm,
                                            const DebugOptions &options) {
  pm.enableVerifier(true);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(mlir::applyPassManagerCLOptions(pm)))
    return mlir::failure();
  if (!options.dumpIRPath.empty()) {
    pm.enableIRPrintingToFileTree(
        [](Pass *, Operation *) { return false; },
        [](Pass *, Operation *) { return true; }, true, false, false,
        options.dumpIRPath, OpPrintingFlags().elideLargeElementsAttrs(32));
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Adhoc Passes
//===----------------------------------------------------------------------===//
namespace {

// This pass executes a "convert-stablehlo-scalar-to-arith" dynamically on all
// functions with the `cluster.host` attribute.
class HloToArithDynamicPipelinePass
    : public PassWrapper<HloToArithDynamicPipelinePass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HloToArithDynamicPipelinePass)

  StringRef getArgument() const override {
    return "stablehlo-to-arith-pipeline";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.isPrivate() || !func->hasAttr("cluster.host"))
      return;

    OpPassManager dynamicPM("func.func");
    dynamicPM.addPass(createConvertStablehloScalarToArithPass());
    if (failed(runPipeline(dynamicPM, func)))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class HloToStdPass
    : public PassWrapper<HloToStdPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HloToStdPass)

  StringRef getArgument() const override { return "stablehlo-to-std"; }
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Apply other preparation and simplification patterns.
    RewritePatternSet patterns(func->getContext());
    // Convert `stablehlo.constant` to `arith.constant`.
    patterns.add<stablehlo::ConstantOp>(
        +[](stablehlo::ConstantOp constOp,
            PatternRewriter &rewriter) -> LogicalResult {
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp,
                                                         constOp.getValue());
          return success();
        });

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      emitError(func.getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// StableHLO Signature Refinement Entrypoint
//===----------------------------------------------------------------------===//

mlirtrt::StatusOr<mlir::FunctionType>
compiler::getStableHLOProgramRefinedSignature(
    CompilerClient &client, mlir::ModuleOp module,
    const StableHLOProgramSignatureRefinementOptions &options) {

#ifndef NDEBUG
  //===----------------------------------------------------------------------===//
  // Set debug options.
  //===----------------------------------------------------------------------===//
  if (options.debugOptions.enableLLVMDebugFlag) {
    SmallVector<const char *> debugTypeLiterals =
        llvm::map_to_vector(options.debugOptions.llvmDebugTypes,
                            [](const std::string &x) { return x.c_str(); });
    llvm::setCurrentDebugTypes(debugTypeLiterals.data(),
                               debugTypeLiterals.size());
    llvm::DebugFlag = true;
  }
#endif

  //===----------------------------------------------------------------------===//
  // Setup pass manager
  //===----------------------------------------------------------------------===//

  mlir::PassManager pm(module->getContext());
  if (failed(setupPassManager(pm, options.debugOptions))) {
    /// TODO: Ignored. This can fail if pass manager static CL options were not
    /// registered/initialized. This happens through invocation of e.g. this
    /// function in e.g. Python bindings or standalone calls to C++ or C API
    /// without doing all the typical static CL setup. We should instead be
    /// accepting a PassManager here that has already been setup to the caller's
    /// specifications.
  }

  // Add pre-processing passes.
  {
    mlir::StableHloInputOptions opts{};
    opts.legalizeControlFlowToSCF = false;
    opts.convertChloToStablehlo = false;
    mlir::buildStablehloPreProcessingPipeline(pm, opts);
  }

  // Run pass pipeline.
  if (mlir::failed(pm.run(module)))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to run compilation pipeline");

  // Get the signature.
  auto func = llvm::dyn_cast_or_null<func::FuncOp>(
      module.lookupSymbol(options.funcName));
  if (!func)
    return getInvalidArgStatus(
        "function with name {0} does not exist in the MLIR module",
        options.funcName);

  return func.getFunctionType();
}

//===----------------------------------------------------------------------===//
// StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

StableHLOToExecutableOptions::StableHLOToExecutableOptions(
    TaskExtensionRegistry extensions)
    : extensions(std::move(extensions)) {
  debugOptions.addToOptions(*this);

  // Link in options for all extensions.
  for (auto &[id, ext] : this->extensions)
    ext->addToOptions(*this);

  addOption(
      "plan-clustering-disallow-host-tensors-in-tensorrt-clusters",
      disallowHostTensorsInTensorRTClusters, llvm::cl::init(false),
      llvm::cl::desc("Don't allow TensorRt clusters to contain host tensor "
                     "calculations (but they can still be inputs)"));
  addOption("executor-index-bitwidth", executorIndexBitwidth,
            llvm::cl::init(64));
  addOption("device-compute-capability", deviceComputeCapability,
            llvm::cl::init(64),
            llvm::cl::desc("Sets the device compute capbility. Only relevant "
                           "if '--device-infer-from-host=false'"));
  addOption("device-max-shared-memory-per-block-kb",
            deviceMaxSharedMemoryPerBlockKb, llvm::cl::init(0));
  addOption("device-infer-from-host", shouldInferDeviceOptionsFromHost,
            llvm::cl::init(true),
            llvm::cl::desc("Infers device information from host"));
  addOption("entrypoint", entrypoint, llvm::cl::init("main"),
            llvm::cl::desc("entrypoint function name"));
}

StableHLOToExecutableOptions &StableHLOToExecutableOptions::setDeviceOptions(
    int64_t computeCapability, int64_t maxSharedMemoryPerBlockKb) {
  deviceMaxSharedMemoryPerBlockKb = maxSharedMemoryPerBlockKb;
  deviceComputeCapability = computeCapability;
  return *this;
}

Status StableHLOToExecutableOptions::inferDeviceOptionsFromHost() {
  cudaDeviceProp properties;
  cudaError_t err = cudaGetDeviceProperties(&properties, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device properties");

  int ccMajor = 0;
  int ccMinor = 0;
  err = cudaDeviceGetAttribute(
      &ccMajor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device compute capability");
  err = cudaDeviceGetAttribute(
      &ccMinor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, 0);
  if (err != cudaSuccess)
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to get cuda device compute capability");

  // We want SM version as a single number.
  int64_t smVersion = ccMajor * 10 + ccMinor;
  this->deviceComputeCapability = smVersion;
  this->deviceMaxSharedMemoryPerBlockKb = properties.sharedMemPerBlock / 1024;
  return Status::getOk();
}

//===----------------------------------------------------------------------===//
// StableHloToExecutableTask
//===----------------------------------------------------------------------===//

static void populateExtensionPasses(
    mlir::OpPassManager &pm, const StableHLOToExecutableOptions &options,
    StableHLOToExecutableOptions::ExtensionBase::Phase phase) {
  for (auto &[key, ext] : options.extensions) {
    llvm::cast<StableHLOToExecutableOptions::ExtensionBase>(ext.get())
        ->populatePasses(pm, phase, options);
  }
}

void StableHloToExecutableTask::buildStablehloClusteringPipeline(
    OpPassManager &pm, const StableHLOToExecutableOptions &opts) {
  using Phase = StableHLOToExecutableOptions::ExtensionBase::Phase;
  pm.addPass(createConvertStablehloToScfPass());

  // Add pre-clustering extension passes
  populateExtensionPasses(pm, opts, Phase::PreClustering);

  plan::StablehloClusteringPassOptions clusteringOpts{};
  clusteringOpts.disallowHostTensorsInTensorRTClusters =
      opts.disallowHostTensorsInTensorRTClusters;
  clusteringOpts.entrypoint = opts.entrypoint;
  plan::buildPlanSegmentationPipeline(pm, clusteringOpts);

  populateExtensionPasses(pm, opts, Phase::PostClustering);

  pm.addPass(createCanonicalizerPass());

  // Compile outlined funcs marked with `cluster.host`. The HLO in these
  // functions should be scalarized.
  pm.addNestedPass<func::FuncOp>(
      std::make_unique<HloToArithDynamicPipelinePass>());

  pm.addNestedPass<func::FuncOp>(std::make_unique<HloToStdPass>());

  pm.addPass(createInlinerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // We then perform some final simplification on the top-level func.func ops
  // (e.g. public entrypoint functions).
  pm.addNestedPass<func::FuncOp>(createSCFDetensorizeLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void StableHloToExecutableTask::buildPostClusteringPipeline(
    OpPassManager &pm, const StableHLOToExecutableOptions &opts) {
  using Phase = StableHLOToExecutableOptions::ExtensionBase::Phase;
  populateExtensionPasses(pm, opts, Phase::PreBufferization);

  // Perform bufferization.
  pm.addPass(createMemRefCastEliminationPass());
  pm.addPass(plan::createPlanAllocTensorsPass());
  pm.addPass(plan::createPlanBufferizePass());
  pm.addPass(createMemRefCastEliminationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  plan::buildPlanBufferOptimizationPipeline(pm);

  populateExtensionPasses(pm, opts, Phase::PostBufferization);

  pm.addPass(createConvertMemRefToCUDAPass());
  pm.addPass(createConvertPlanToExecutorPass());
  pm.addPass(executor::createExecutorAllocsToGlobalsPass());
  pm.addNestedPass<func::FuncOp>(
      executor::createExecutorPopulateFunctionMetadataPass());

  populateExtensionPasses(pm, opts, Phase::ExecutorLowering);

  ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
  cudaToExecutorOpts.indexBitwidth = opts.executorIndexBitwidth;
  cudaToExecutorOpts.usePackedMemRefCConv = opts.executorUsePackedMemRefCConv;
  pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));

  pm.addPass(createDropNestedModulesPass());
}

void StableHloToExecutableTask::populatePassManager(
    mlir::PassManager &pm, const StableHLOToExecutableOptions &options) {
  if (failed(setupPassManager(pm, options.debugOptions))) {
    /// TODO: Ignored. This can fail if pass manager static CL options were not
    /// registered/initialized. This happens through invocation of e.g. this
    /// function in e.g. Python bindings or standalone calls to C++ or C API
    /// without doing all the typical static CL setup. We should instead be
    /// accepting a PassManager here that has already been setup to the caller's
    /// specifications.
  }

  // StableHLO Preprocessing
  mlir::StableHloInputOptions opts{};
  opts.legalizeControlFlowToSCF = false;
  opts.convertChloToStablehlo = false;
  mlir::buildStablehloPreProcessingPipeline(pm, opts);

  buildStablehloClusteringPipeline(pm, options);

  buildPostClusteringPipeline(pm, options);

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth = options.executorIndexBitwidth;
  stdToExecOpts.usePackedMemRefCConv = true;
  mlir::executor::buildExecutorLoweringPipeline(pm, stdToExecOpts);
}

StatusOr<std::unique_ptr<runtime::Executable>>
StableHloToExecutableTask::compileStableHLOToExecutable(
    mlir::ModuleOp module, const StableHLOToExecutableOptions &options) {
  LLVM_DEBUG({
    DBGS() << "compiling with options:\n";
    options.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

#ifndef NDEBUG
  //===----------------------------------------------------------------------===//
  // Set debug options.
  //===----------------------------------------------------------------------===//
  if (options.debugOptions.enableLLVMDebugFlag) {
    SmallVector<const char *> debugTypeLiterals =
        llvm::map_to_vector(options.debugOptions.llvmDebugTypes,
                            [](const std::string &x) { return x.c_str(); });
    llvm::setCurrentDebugTypes(debugTypeLiterals.data(),
                               debugTypeLiterals.size());
    llvm::DebugFlag = true;
  }
#endif

  //===----------------------------------------------------------------------===//
  // Setup pass manager
  //===----------------------------------------------------------------------===//

  StableHloToExecutableTask runner(module->getContext(), options);
  if (failed(setupPassManager(runner, options.debugOptions))) {
    /// TODO: Ignored. This can fail if pass manager static CL options were not
    /// registered/initialized. This happens through invocation of e.g. this
    /// function in e.g. Python bindings or standalone calls to C++ or C API
    /// without doing all the typical static CL setup. We should instead be
    /// accepting a PassManager here that has already been setup to the caller's
    /// specifications.
  }
  if (failed(runner.run(module)))
    return getInternalErrorStatus(
        "failed to run compilation on module with symbol name: {0}",
        module.getName() ? *module.getName() : "no-symbol-name");

  //===----------------------------------------------------------------------===//
  // Translate to Runtime Executable
  //===----------------------------------------------------------------------===//

  FailureOr<std::unique_ptr<runtime::ExecutableStorage>> exeStorage =
      mlir::translateToRuntimeExecutable(module);
  if (failed(exeStorage))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to translate compiled MLIR module to a "
                            "MLIR-TensorRT runtime Executable");

#ifndef NDEBUG
  // Turn debugging back off if we turned it on.
  if (options.debugOptions.enableLLVMDebugFlag)
    llvm::DebugFlag = false;
#endif

  return std::make_unique<runtime::Executable>(std::move(*exeStorage));
}

mlirtrt::StatusOr<std::unique_ptr<runtime::Executable>>
StableHloToExecutableTask::compileStableHLOToExecutable(
    CompilerClient &client, mlir::ModuleOp module,
    const StableHLOToExecutableOptions &options) {
  if (client.getContext() != module->getContext())
    return getInternalErrorStatus("CompilerClient has a MLIRContext that is "
                                  "different from the ModuleOp's MLIRContext");

  LLVM_DEBUG({
    DBGS() << "compiling with options:\n";
    options.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

#ifndef NDEBUG
  if (options.debugOptions.enableLLVMDebugFlag) {
    SmallVector<const char *> debugTypeLiterals =
        llvm::map_to_vector(options.debugOptions.llvmDebugTypes,
                            [](const std::string &x) { return x.c_str(); });
    llvm::setCurrentDebugTypes(debugTypeLiterals.data(),
                               debugTypeLiterals.size());
    llvm::DebugFlag = true;
  }
#endif

  mlir::PassManager &runner =
      client.getOrCreatePassManager<StableHloToExecutableTask>(options);

  // Setup pass manager
  if (failed(runner.run(module)))
    return getInternalErrorStatus(
        "failed to run compilation on module with symbol name: {0}",
        module.getName() ? *module.getName() : "no-symbol-name");

  // Translate to Runtime Executable
  FailureOr<std::unique_ptr<runtime::ExecutableStorage>> exeStorage =
      mlir::translateToRuntimeExecutable(module);
  if (failed(exeStorage))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to translate compiled MLIR module to a "
                            "MLIR-TensorRT runtime Executable");

#ifndef NDEBUG
  // Turn debugging back off if we turned it on.
  if (options.debugOptions.enableLLVMDebugFlag)
    llvm::DebugFlag = false;
#endif

  return std::make_unique<runtime::Executable>(std::move(*exeStorage));
}

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

namespace {
struct ClusteringPipelineCliOpts
    : public PassPipelineOptions<ClusteringPipelineCliOpts> {
  Option<bool> lowerStablehloControlFlow{
      *this, "lower-stablehlo-control-flow",
      llvm::cl::desc("lower control flow to scf"), llvm::cl::init(true)};
  Option<int64_t> deviceComputeCapability{
      *this, "device-compute-capability",
      llvm::cl::desc("target device compute capability (SM version)"),
      llvm::cl::init(60)};
  Option<int64_t> deviceMaxSharedMemoryPerBlockKb{
      *this, "device-max-smem-per-block",
      llvm::cl::desc("max shared memory per block (in kilobytes)"),
      llvm::cl::init(50)};
  Option<bool> inferDeviceOptionsFromHost{
      *this, "infer-device-opts-from-host",
      llvm::cl::desc(
          "whether to infer 'device-*' options from host system GPU"),
      llvm::cl::init(true)};
  Option<std::string> entrypoint{*this, "entrypoint", llvm::cl::init(""),
                                 llvm::cl::desc("entrypoint function name")};
};
} // namespace

/// Convert a `ClusteringPipelineCliOpts` into a
/// `StablehloClusteringPipelineOpts`.
static StableHLOToExecutableOptions populateStablehloClusteringPipelineOpts(
    const ClusteringPipelineCliOpts &cliOpts) {
  // Load a default extension set since we don't have access to MLIRContext at
  // this point.
  TaskExtensionRegistry extensions;
  extensions.getOrCreateExtension<StableHLOToExecutableTensorRTExtension>();

  StableHLOToExecutableOptions opts(std::move(extensions));
  opts.deviceComputeCapability = cliOpts.deviceComputeCapability;
  opts.deviceMaxSharedMemoryPerBlockKb =
      cliOpts.deviceMaxSharedMemoryPerBlockKb;
  opts.shouldInferDeviceOptionsFromHost = cliOpts.inferDeviceOptionsFromHost;
  opts.entrypoint = cliOpts.entrypoint;
  return opts;
}

void mlirtrt::compiler::registerStablehloClusteringPipelines() {
  PassRegistration<HloToStdPass>();
  PassRegistration<HloToArithDynamicPipelinePass>();

  PassPipelineRegistration<ClusteringPipelineCliOpts>(
      "stablehlo-clustering-pipeline",
      "apply clustering and initial transformations to stablehlo IR",
      [](OpPassManager &pm, const ClusteringPipelineCliOpts &opts) {
        StableHloToExecutableTask::buildStablehloClusteringPipeline(
            pm, populateStablehloClusteringPipelineOpts(opts));
      });

  PassPipelineRegistration<ClusteringPipelineCliOpts>(
      "post-clustering-pipeline", "apply compilation post-clustering",
      [](OpPassManager &pm, const ClusteringPipelineCliOpts &opts) {
        StableHLOToExecutableOptions finalOpts =
            populateStablehloClusteringPipelineOpts(opts);
        StableHloToExecutableTask::buildPostClusteringPipeline(pm, finalOpts);
      });
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlirtrt::compiler::StableHloToExecutableTask)

#else

StatusOr<std::unique_ptr<runtime::Executable>>
compiler::compileStableHLOToExecutable(
    CompilerClient &client, mlir::ModuleOp module,
    const StableHLOToExecutableOptions &options) {
  return getStatusWithMsg(
      StatusCode::Unimplemented,
      "MLIR-TensorRT was not compiled with StableHLO support");
}

StatusOr<mlir::FunctionType> compiler::getStableHLOProgramRefinedSignature(
    CompilerClient &client, mlir::ModuleOp module,
    const StableHLOProgramSignatureRefinementOptions &options) {
  return getStatusWithMsg(
      StatusCode::Unimplemented,
      "MLIR-TensorRT was not compiled with StableHLO support");
}

#endif // MLIR_TRT_ENABLE_HLO
