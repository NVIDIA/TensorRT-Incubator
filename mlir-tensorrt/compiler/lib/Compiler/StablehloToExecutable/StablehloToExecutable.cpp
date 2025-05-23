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
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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
// StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

StablehloToExecutableOptions::StablehloToExecutableOptions(
    TaskExtensionRegistry extensions, bool enableDebugOptions)
    : CompilationTaskOptions(enableDebugOptions),
      extensions(std::move(extensions)) {
  for (const auto &[typeID, builder] : this->extensions.builders) {
    this->extensions.extensions[typeID] = builder(*this);
  }
}

static TaskExtensionRegistry getDefaultExtensions() {
  TaskExtensionRegistry extensions;
  extensions.registerExtension<StablehloToExecutableTensorRTExtension>();
  return extensions;
}

StablehloToExecutableOptions::StablehloToExecutableOptions(
    bool enableDebugOptions)
    : mlirtrt::compiler::StablehloToExecutableOptions(getDefaultExtensions(),
                                                      enableDebugOptions) {}

//===----------------------------------------------------------------------===//
// StableHloToExecutableTask
//===----------------------------------------------------------------------===//

StablehloToExecutableTask::StablehloToExecutableTask(
    MLIRContext *ctx, std::unique_ptr<StablehloToExecutableOptions> options)
    : CompilationTask(ctx, std::move(options)) {}

static void populateExtensionPasses(
    mlir::OpPassManager &pm, const StablehloToExecutableOptions &options,
    StablehloToExecutableOptions::ExtensionBase::Phase phase) {
  for (auto &[key, ext] : options.extensions) {
    llvm::cast<StablehloToExecutableOptions::ExtensionBase>(ext.get())
        ->populatePasses(pm, phase, options);
  }
}

void StablehloToExecutableTask::buildClusteringPipeline(
    OpPassManager &pm, const StablehloToExecutableOptions &opts) {
  using Phase = StablehloToExecutableOptions::ExtensionBase::Phase;
  pm.addPass(createConvertStablehloToScfPass());

  // Add pre-clustering extension passes
  populateExtensionPasses(pm, opts, Phase::PreClustering);

  plan::ClusteringPassOptions clusteringOpts{};
  clusteringOpts.entrypoint = opts.entrypoint;
  clusteringOpts.inputKind = plan::InputKind::Stablehlo;
  plan::buildPlanSegmentationPipeline(pm, clusteringOpts);

  // Compile outlined scalarizable host clusters.
  pm.addNestedPass<func::FuncOp>(createProcessStablehloHostClustersPass());
  pm.addNestedPass<func::FuncOp>(createConvertStablehloConstantsToArithPass());

  populateExtensionPasses(pm, opts, Phase::PostClustering);

  pm.addNestedPass<func::FuncOp>(plan::createPostClusteringValidationPass());

  // We then perform some final simplification on the top-level func.func ops
  // (e.g. public entrypoint functions).
  pm.addNestedPass<func::FuncOp>(createSCFDetensorizeLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void StablehloToExecutableTask::buildPostClusteringPipeline(
    OpPassManager &pm, const StablehloToExecutableOptions &opts) {
  using Phase = StablehloToExecutableOptions::ExtensionBase::Phase;
  populateExtensionPasses(pm, opts, Phase::PreBufferization);

  // Perform bufferization.
  plan::buildPlanBufferizationPipeline(
      pm, {opts.get<PlanAllocOptions>().forceEntrypointsReturnAllocs},
      bufferization::DeallocationOptions{
          /*privateFuncDynamicOwnership=*/false});

  populateExtensionPasses(pm, opts, Phase::PostBufferization);

  pm.addPass(createConvertMemRefToCUDAPass());

  HostTarget hostTarget = opts.hostTarget;
  if (hostTarget == HostTarget::Executor) {
    pm.addPass(createConvertPlanToExecutorPass());
    pm.addPass(executor::createExecutorAllocsToGlobalsPass());
    pm.addNestedPass<func::FuncOp>(
        executor::createExecutorPopulateFunctionMetadataPass());
  }

  populateExtensionPasses(pm, opts, Phase::ExecutorLowering);

  if (hostTarget == HostTarget::Executor) {
    ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
    cudaToExecutorOpts.indexBitwidth =
        opts.get<ExecutorOptions>().indexBitwidth;
    cudaToExecutorOpts.usePackedMemRefCConv =
        opts.get<ExecutorOptions>().usePackedMemRefCConv;
    pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));
  }

  pm.addPass(createDropNestedModulesPass());
}

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void StablehloToExecutableTask::populatePassManager(
    mlir::OpPassManager &pm, const StablehloToExecutableOptions &options) {
  pm.addPass(createPopulateDefaultBackendMetadataPass(
      PopulateDefaultBackendMetadataPassOptions{
          options.disallowHostTensorsInTensorRTClusters, NV_TENSORRT_MAJOR}));

  // StableHLO Preprocessing
  mlirtrt::compiler::StableHloInputOptions opts{};
  opts.legalizeControlFlowToSCF = false;
  opts.preserveChloErf = true;
  opts.preserveChloTopK = true;
  mlirtrt::compiler::buildStablehloPreProcessingPipeline(pm, opts);

  buildClusteringPipeline(pm, options);

  buildPostClusteringPipeline(pm, options);

  HostTarget hostTarget = options.hostTarget;
  if (hostTarget == HostTarget::Executor) {
    mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
    stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
    stdToExecOpts.usePackedMemRefCConv = true;
    mlir::executor::buildExecutorLoweringPipeline(pm, stdToExecOpts);
    return;
  }

  // LLVM and EmitC targets will execute a common set of passes except for the
  // SCF-to-CF pass.
  if (hostTarget == HostTarget::LLVM || hostTarget == HostTarget::EmitC) {
    pm.addPass(createConvertComplexToStandardPass());

    // For EmitC lowering, we rely on preserving control flow. Otherwise the C
    // code could be very unreadable.
    if (hostTarget != HostTarget::EmitC)
      pm.addPass(createConvertSCFToCFPass());

    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(memref::createExpandOpsPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    addCleanupPasses(pm);
    pm.addPass(affine::createAffineExpandIndexOpsPass());
    pm.addPass(mlir::createLowerAffinePass());
    addCleanupPasses(pm);

    if (hostTarget == HostTarget::LLVM) {
      pm.addPass(LLVM::createRequestCWrappersPass());
      ConvertCUDAToLLVMPassOptions cudaToLLVMOpts;
      cudaToLLVMOpts.artifactsDirectory = options.artifactsDirectory;
      pm.addPass(createConvertCUDAToLLVMPass(std::move(cudaToLLVMOpts)));
      pm.addPass(createConvertHostToLLVMPass());

      // Generally there is a lot of cleanup that can happen here after creation
      // of LLVM IR (e.g. insert|extract forwarding).
      addCleanupPasses(pm);
    }

    // For EmitC, just run Host-to-EmitC followed
    // by cleanup and expression forming.
    if (hostTarget == HostTarget::EmitC) {
      pm.addPass(createConvertHostToEmitCPass({options.artifactsDirectory}));
      addCleanupPasses(pm);
      // The EmitC "form-expressions" pass combines operations into
      // "expression regions" where possible, which allows the C++ translation
      // to be more concise.
      pm.addPass(emitc::createFormExpressionsPass());
    }
    return;
  }
}

mlirtrt::StatusOr<std::unique_ptr<runtime::Executable>>
StablehloToExecutableTask::compileStableHLOToExecutable(
    CompilerClient &client, mlir::ModuleOp module,
    const StablehloToExecutableOptions &options) {
  if (client.getContext() != module->getContext())
    return getInternalErrorStatus("CompilerClient has a MLIRContext that is "
                                  "different from the ModuleOp's MLIRContext");

  LLVM_DEBUG({
    DBGS() << "compiling with options:\n";
    options.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

#ifndef NDEBUG
  if (options.get<DebugOptions>().enableLLVMDebugFlag) {
    SmallVector<const char *> debugTypeLiterals =
        llvm::map_to_vector(options.get<DebugOptions>().llvmDebugTypes,
                            [](const std::string &x) { return x.c_str(); });
    llvm::setCurrentDebugTypes(debugTypeLiterals.data(),
                               debugTypeLiterals.size());
    llvm::DebugFlag = true;
  }
#endif

  std::string result;
  llvm::raw_string_ostream ss(result);
  options.print(ss);
  ss.flush();
  StatusOr<CompilationTaskBase *> runner =
      client.getCompilationTask<StablehloToExecutableTask>(
          llvm::StringRef(result).drop_front(1).drop_back(1),
          /*enableDebugOptions=*/false);
  if (!runner.isOk())
    return runner.getStatus();

  // Setup pass manager
  if (failed((*runner)->run(module)))
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
  if (options.get<DebugOptions>().enableLLVMDebugFlag)
    llvm::DebugFlag = false;
#endif

  return std::make_unique<runtime::Executable>(std::move(*exeStorage));
}

void mlirtrt::compiler::registerStableHloToExecutableTask() {
  registerCompilationTask<StablehloToExecutableTask>(
      "stablehlo-to-executable",
      [](CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options,
         bool enableDebugOptions) -> StatusOr<CompilationTaskBase *> {
        // Load available extensions.
        mlir::MLIRContext *context = client.getContext();
        mlir::plan::PlanDialect *planDialect =
            context->getLoadedDialect<mlir::plan::PlanDialect>();
        compiler::TaskExtensionRegistry extensions =
            planDialect->extensionConstructors
                .getExtensionRegistryForTask<StablehloToExecutableTask>();

        auto opts = std::make_unique<StablehloToExecutableOptions>(
            std::move(extensions), enableDebugOptions);

        std::string err;
        if (failed(opts->parse(options, err)))
          return getInvalidArgStatus(
              "failed to parse options string \"{0:$[ ]}\" due to error {1}",
              llvm::iterator_range(options), err);

        std::optional<llvm::hash_code> hashCode = opts->getHash();
        if (!hashCode)
          return getInvalidArgStatus("failed to hash options");

        CompilationTaskBase *cached = client.lookupCachedCompilationTask(
            mlir::TypeID::get<StablehloToExecutableTask>(), *hashCode);
        if (cached)
          return cached;

        auto newPM = std::make_unique<StablehloToExecutableTask>(
            client.getContext(), std::move(opts));
        auto ptr = newPM.get();
        client.updateCachedCompilationTask<StablehloToExecutableTask>(
            *hashCode, std::move(newPM));
        return ptr;
      });
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlirtrt::compiler::StablehloToExecutableTask)

#endif // MLIR_TRT_ENABLE_HLO
