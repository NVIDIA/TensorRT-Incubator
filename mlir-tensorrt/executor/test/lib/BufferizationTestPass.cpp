//===- BufferizationTestPass.cpp ------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// A bufferization pass used just for integration tests.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::executor;

namespace {
class ExecutorBufferizationTestPass
    : public PassWrapper<ExecutorBufferizationTestPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExecutorBufferizationTestPass)

  StringRef getArgument() const override {
    return "test-executor-one-shot-bufferize";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    assert(ctx == module->getContext() && "MLIRContexts are not the same");
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocsFromLoops = false;
    options.defaultMemorySpaceFn =
        [ctx](TensorType t) -> std::optional<Attribute> {
      return executor::MemoryTypeAttr::get(ctx, MemoryType::host);
    };
    options.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::InferLayoutMap);

    if (failed(bufferization::runOneShotModuleBufferize(module, options))) {
      emitError(module.getLoc()) << "failed to bufferize module";
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::executor {
void registerTestExecutorBufferizePass() {
  PassRegistration<ExecutorBufferizationTestPass>();

  PassPipelineRegistration<> executorBufferizationPipeline(
      "test-executor-bufferization-pipeline",
      "Run one-shot-bufferization and buffer deallocation pipelines",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<ExecutorBufferizationTestPass>());
        pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
        bufferization::BufferDeallocationPipelineOptions deallocOptions{};
        bufferization::buildBufferDeallocationPipeline(pm, deallocOptions);
        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());
      });
}
} // namespace mlir::executor
