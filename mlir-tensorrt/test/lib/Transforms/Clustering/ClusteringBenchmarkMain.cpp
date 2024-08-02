//===- ClusteringBenchmarkTests.cpp ---------------------------------------===//
//
// Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Tests that benchmark the clustering transform.
///
//===----------------------------------------------------------------------===//
#include "benchmark/benchmark.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Registration/RegisterMlirTensorRtDialects.h"
#include "mlir-tensorrt/Registration/RegisterMlirTensorRtPasses.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::plan;
namespace cl = llvm::cl;

static cl::opt<std::string> inputSourceFile("source", cl::desc("Source file"),
                                            cl::Required);

auto BM_test = [](benchmark::State &state, MLIRContext *ctx) {
  mlir::PassManager pm(ctx);
  pm.enableVerifier(true);
  pm.enableCrashReproducerGeneration("crash-reproducer.mlir",
                                     /*genLocalReproducer=*/false);
  plan::StablehloClusteringPassOptions opts{};
  opts.entrypoint = "main";
  pm.addPass(plan::createStablehloClusteringPass(opts));
  pm.addPass(plan::createOutlineClustersPass());

  mlir::ParserConfig config(ctx);
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceFile<ModuleOp>(inputSourceFile, config);

  for (auto _ : state) {
    OwningOpRef<ModuleOp> clone = cast<ModuleOp>((*module)->clone());
    if (failed(pm.run(*clone))) {
      llvm_unreachable("failed to run pass manager");
    }
  }
};

int main(int argc, char *argv[]) {
  DialectRegistry registry;
  registerAllMlirTensorRtDialects(registry);
  tensorrt::registerAllMlirTensorRtPasses();
  MLIRContext context(registry);
  benchmark::RegisterBenchmark("clustering_test", BM_test, &context);
  benchmark::Initialize(&argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}