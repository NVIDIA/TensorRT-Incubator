//===- GenerateSortTests.cpp ----------------------------------------------===//
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
/// Unit tests for GPU merge sort kernel generation.
///
/// This file contains unit tests that programmatically generate GPU merge sort
/// kernels using the MergeSortKernelGenerator API. The tests verify:
///
/// 1. Correct generation of the merge sort kernel IR (block sort, partition,
///    and merge kernels)
/// 2. Generation of a complete executable test program that exercises the
///    generated kernels across multiple array sizes
///
/// The generated MLIR follows the CUB (CUDA Unbound) DeviceMergeSort algorithm
/// with a two-stage approach:
///   - Stage 1: Block-level sorting using bitonic sort within thread blocks
///   - Stage 2: Multi-pass global merging with partition and merge kernels
///
/// The GenerateExecutableSortProgramTest outputs a complete MLIR file
/// (merge-sort-test-original.mlir) that can be compiled and executed to verify
/// the correctness of the generated kernels on actual GPU hardware.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-kernel/InitAllDialects.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/GenerateSort.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::kernel;

class GenerateSortTest : public ::testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    kernel::registerAllRequiredDialects(registry);
    registry.insert<executor::ExecutorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  MLIRContext context;
};

TEST_F(GenerateSortTest, CreateGPUModuleTest) {
  OpBuilder builder(&context);
  auto module = ModuleOp::create(UnknownLoc::get(&context));
  builder.setInsertionPointToStart(module.getBody());
  auto gpuModule =
      builder.create<gpu::GPUModuleOp>(UnknownLoc::get(&context), "merge_sort");
  builder.setInsertionPointAfter(gpuModule);

  SymbolTableCollection symbolTables;

  // Use the public API to create kernels, which internally creates the GPU
  // module
  Type i32Type = IntegerType::get(&context, 32);
  MergeSortConfig config;
  config.keysOnly = true;

  auto result = MergeSortKernelGenerator::createMergeSortKernels(
      builder, module.getLoc(), i32Type, Type{}, module, gpuModule,
      symbolTables, config);

  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(mlir::verify(module).succeeded());
}

TEST_F(GenerateSortTest, GenerateExecutableSortProgramTest) {

  OpBuilder builder(&context);
  auto module = ModuleOp::create(UnknownLoc::get(&context));
  builder.setInsertionPointToStart(module.getBody());
  auto gpuModule =
      builder.create<gpu::GPUModuleOp>(UnknownLoc::get(&context), "merge_sort");
  builder.setInsertionPointAfter(gpuModule);

  SymbolTableCollection symbolTables;

  // Test configuration - tests multiple sizes
  std::vector<int64_t> testSizes = {16, 128, 256, 512, 1024, 2048};

  Type i32Type = IntegerType::get(&context, 32);

  // Configure merge sort
  MergeSortConfig config;
  config.keysOnly = true;
  config.blockThreads = 128;
  config.itemsPerThread = 4;

  // Create the merge sort kernels (shared across all tests)
  auto result = MergeSortKernelGenerator::createMergeSortKernels(
      builder, module.getLoc(), i32Type, Type{}, module, gpuModule,
      symbolTables, config);
  ASSERT_TRUE(succeeded(result));
  auto kernels = *result;

  // Verify the module
  auto verifyResult = mlir::verify(module.getOperation());
  if (failed(verifyResult)) {
    // Dump the module if verification fails
    llvm::errs() << "Module verification failed. Dumping module:\n";
    module->print(llvm::errs());
    llvm::errs() << "\n";
  }
  EXPECT_TRUE(succeeded(verifyResult));

  // Write to file with test harness in textual MLIR format
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile("merge-sort-test-original.mlir");
  if (of) {
    // Add test runner comments at the top
    of->os() << "// REQUIRES: host-has-at-least-1-gpus\n";
    of->os() << "// RUN: mlir-tensorrt-compiler %s --input=linalg --opts=\"\" "
                "-o - | \\\n";
    of->os() << "// RUN: mlir-tensorrt-runner -input-type=rtexe "
                "-features=core,cuda 2>&1 | FileCheck %s\n\n";
    of->os()
        << "// NOTE: This file is auto-generated by GenerateSortTests.cpp.\n";
    of->os() << "// To regenerate this file, run:\n";
    of->os() << "//   cd /workspaces/mlir-tensorrt\n";
    of->os() << "//   source .env\n";
    of->os() << "//   ./build/kernel/test/Unit/MLIRKernelUtilsTests "
                "--gtest_filter=GenerateSortTest."
                "GenerateExecutableSortProgramTest\n";
    of->os() << "//\n";
    of->os() << "// The test will output this file to the current working "
                "directory.\n";
    of->os() << "// Then copy it to the desired location:\n";
    of->os() << "//   cp merge-sort-test-original.mlir "
                "compiler/test-internal/IntegrationTests/CodeGen/BlockLevel/"
                "merge-sort.mlir\n\n";
    of->os() << "// kernel.sort: Merge sort implementation for GPU\n";
    of->os() << "// Schedule:\n";
    of->os() << "//  This implementation uses a CUB-inspired two-stage GPU "
                "merge sort.\n";
    of->os() << "//  Reference: "
                "https://nvidia.github.io/cccl/cub/api/"
                "structcub_1_1DeviceMergeSort.html\n";
    of->os() << "//\n";
    of->os() << "//\n";
    of->os() << "//  Stage 1 - Block Sort:\n";
    of->os() << "//    Each block sorts 512 elements (128 threads × 4 "
                "items/thread):\n";
    of->os() << "//    - Load elements in striped pattern to registers\n";
    of->os() << "//    - Bitonic sort within each thread's elements\n";
    of->os() << "//    - Block-level merges using shared memory\n";
    of->os() << "//\n";
    of->os() << "//  Stage 2 - Multi-Pass Global Merge:\n";
    of->os() << "//    - Partition kernel computes merge path boundaries\n";
    of->os() << "//    - Merge kernel combines sorted sequences\n";
    of->os() << "//    - Ping-pong buffers alternate between passes\n";
    of->os() << "//    - log2(num_blocks) passes to fully sort\n";
    of->os() << "//\n";
    of->os() << "//  Parameters: BLOCK_THREADS=128, ITEMS_PER_THREAD=4, "
                "TILE_SIZE=512\n";
    of->os() << "//  Shared memory: 2KB (block sort), 4KB (merge)\n";
    of->os() << "//\n";
    of->os() << "//  Test Coverage:\n";
    of->os() << "//    Array sizes: ";
    for (size_t i = 0; i < testSizes.size(); i++) {
      of->os() << testSizes[i];
      if (i < testSizes.size() - 1)
        of->os() << ", ";
    }
    of->os() << "\n";
    of->os()
        << "//    Tests single block, two blocks, and multiple merge passes\n";
    of->os() << "\n";

    // Add FileCheck patterns for each size
    for (auto size : testSizes) {
      of->os() << "// CHECK: === Testing Array Size: " << size << " ===\n";
      of->os() << "// CHECK: Input - First 10: \n";
      if (size > 10)
        of->os() << "// CHECK: Input - Last 10: \n";
      of->os() << "// CHECK: Output - First 10: \n";
      if (size > 10)
        of->os() << "// CHECK: Output - Last 10: \n";
      of->os() << "// CHECK: Sort violations found: 0\n";
    }
    of->os() << "\n";

    // Add the gpu.container_module attribute to the module
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(&context));

    // Add the main test function to the module before printing
    OpBuilder moduleBuilder(&context);
    moduleBuilder.setInsertionPointToEnd(module.getBody());
    Location loc = UnknownLoc::get(&context);

    auto mainFuncType =
        moduleBuilder.getFunctionType({}, moduleBuilder.getI32Type());
    auto mainFunc =
        moduleBuilder.create<func::FuncOp>(loc, "main", mainFuncType);
    Block *entryBlock = mainFunc.addEntryBlock();
    moduleBuilder.setInsertionPointToStart(entryBlock);

    // Constants
    Value c0 = moduleBuilder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = moduleBuilder.create<arith::ConstantIndexOp>(loc, 1);
    Value c0_i32 = moduleBuilder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value c1_i32 = moduleBuilder.create<arith::ConstantIntOp>(loc, 1, 32);

    // Helper lambda to generate a test for one array size
    auto generateTestForSize = [&](int64_t ARRAY_SIZE) {
      // Print empty line first
      moduleBuilder.create<executor::PrintOp>(loc, ValueRange{},
                                              moduleBuilder.getStringAttr(""));
      // Print test label
      std::string labelMsg =
          "=== Testing Array Size: " + std::to_string(ARRAY_SIZE) + " ===";
      moduleBuilder.create<executor::PrintOp>(
          loc, ValueRange{}, moduleBuilder.getStringAttr(labelMsg));

      Value numElements =
          moduleBuilder.create<arith::ConstantIndexOp>(loc, ARRAY_SIZE);
      auto hostTensorType = RankedTensorType::get({ARRAY_SIZE}, i32Type);
      Value emptyTensor = moduleBuilder.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{ARRAY_SIZE}, i32Type, ValueRange{});

      Value hostTensor =
          moduleBuilder
              .create<scf::ForOp>(
                  loc, c0, numElements, c1, ValueRange{emptyTensor},
                  [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                      ValueRange iterArgs) {
                    Value currentTensor = iterArgs[0];
                    // Create unsorted data using a pattern: (ARRAY_SIZE - i -
                    // 1) XOR (i
                    // * 7) This creates a pseudo-random pattern that isn't
                    // sorted
                    Value valIdx = nestedBuilder.create<arith::SubIOp>(
                        loc, numElements, iv);
                    Value valIdxMinusOne =
                        nestedBuilder.create<arith::SubIOp>(loc, valIdx, c1);
                    Value val = nestedBuilder.create<arith::IndexCastOp>(
                        loc, i32Type, valIdxMinusOne);

                    // Add some scrambling to make it unsorted
                    Value seven_i32 =
                        nestedBuilder.create<arith::ConstantIntOp>(loc, 7, 32);
                    Value ivAsI32 = nestedBuilder.create<arith::IndexCastOp>(
                        loc, i32Type, iv);
                    Value ivTimesSeven = nestedBuilder.create<arith::MulIOp>(
                        loc, ivAsI32, seven_i32);
                    Value scrambledVal = nestedBuilder.create<arith::XOrIOp>(
                        loc, val, ivTimesSeven);

                    Value insertedTensor =
                        nestedBuilder.create<tensor::InsertOp>(
                            loc, scrambledVal, currentTensor, ValueRange{iv});
                    nestedBuilder.create<scf::YieldOp>(
                        loc, ValueRange{insertedTensor});
                  })
              .getResult(0);

      // Print first 10 elements of input using scf.for
      moduleBuilder.create<executor::PrintOp>(
          loc, ValueRange{}, moduleBuilder.getStringAttr("Input - First 10: "));
      Value endFirst = moduleBuilder.create<arith::ConstantIndexOp>(
          loc, std::min(ARRAY_SIZE, int64_t(10)));
      moduleBuilder.create<scf::ForOp>(
          loc, c0, endFirst, c1, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            Value elem = nestedBuilder.create<tensor::ExtractOp>(
                loc, hostTensor, ValueRange{iv});
            nestedBuilder.create<executor::PrintOp>(
                loc, ValueRange{elem}, nestedBuilder.getStringAttr("%d "));
            nestedBuilder.create<scf::YieldOp>(loc, ValueRange{});
          });
      moduleBuilder.create<executor::PrintOp>(loc, ValueRange{},
                                              moduleBuilder.getStringAttr(""));

      // Print last 10 elements of input using scf.for
      if (ARRAY_SIZE > 10) {
        moduleBuilder.create<executor::PrintOp>(
            loc, ValueRange{},
            moduleBuilder.getStringAttr("Input - Last 10: "));
        Value startLast = moduleBuilder.create<arith::ConstantIndexOp>(
            loc, std::max(int64_t(0), ARRAY_SIZE - 10));
        moduleBuilder.create<scf::ForOp>(
            loc, startLast, numElements, c1, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                ValueRange iterArgs) {
              Value elem = nestedBuilder.create<tensor::ExtractOp>(
                  loc, hostTensor, ValueRange{iv});
              nestedBuilder.create<executor::PrintOp>(
                  loc, ValueRange{elem}, nestedBuilder.getStringAttr("%d "));
              nestedBuilder.create<scf::YieldOp>(loc, ValueRange{});
            });
        moduleBuilder.create<executor::PrintOp>(
            loc, ValueRange{}, moduleBuilder.getStringAttr(""));
      }

      // Create device tensor
      Value deviceTensor = moduleBuilder.create<bufferization::AllocTensorOp>(
          loc, hostTensorType, /*dynamicSizes=*/ValueRange{},
          /*copy=*/hostTensor);

      // Cast to dynamic for dispatch call
      auto dynamicTensorType =
          RankedTensorType::get({ShapedType::kDynamic}, i32Type);
      Value deviceTensorDynamic = moduleBuilder.create<tensor::CastOp>(
          loc, dynamicTensorType, deviceTensor);

      SmallVector<Value> args = {deviceTensorDynamic, numElements};
      auto dispatchCall = moduleBuilder.create<func::CallOp>(
          loc, kernels.dispatchFunc.getName(),
          TypeRange{dynamicTensorType}, // Returns sorted tensor<?xi32>
          args);
      Value sortedDevice = dispatchCall.getResult(0);

      // Cast back to static for verification
      Value sortedDeviceStatic = moduleBuilder.create<tensor::CastOp>(
          loc, hostTensorType, sortedDevice);

      // Copy back to host tensor
      Value sortedHost = moduleBuilder.create<bufferization::AllocTensorOp>(
          loc, hostTensorType, /*dynamicSizes=*/ValueRange{},
          /*copy=*/sortedDeviceStatic);

      // Print first 10 elements of output using scf.for
      moduleBuilder.create<executor::PrintOp>(
          loc, ValueRange{},
          moduleBuilder.getStringAttr("Output - First 10: "));
      Value endFirstOut = moduleBuilder.create<arith::ConstantIndexOp>(
          loc, std::min(ARRAY_SIZE, int64_t(10)));
      moduleBuilder.create<scf::ForOp>(
          loc, c0, endFirstOut, c1, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            Value elem = nestedBuilder.create<tensor::ExtractOp>(
                loc, sortedHost, ValueRange{iv});
            nestedBuilder.create<executor::PrintOp>(
                loc, ValueRange{elem}, nestedBuilder.getStringAttr("%d "));
            nestedBuilder.create<scf::YieldOp>(loc, ValueRange{});
          });
      moduleBuilder.create<executor::PrintOp>(loc, ValueRange{},
                                              moduleBuilder.getStringAttr(""));

      // Print last 10 elements of output using scf.for
      if (ARRAY_SIZE > 10) {
        moduleBuilder.create<executor::PrintOp>(
            loc, ValueRange{},
            moduleBuilder.getStringAttr("Output - Last 10: "));
        Value startLastOut = moduleBuilder.create<arith::ConstantIndexOp>(
            loc, std::max(int64_t(0), ARRAY_SIZE - 10));
        moduleBuilder.create<scf::ForOp>(
            loc, startLastOut, numElements, c1, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                ValueRange iterArgs) {
              Value elem = nestedBuilder.create<tensor::ExtractOp>(
                  loc, sortedHost, ValueRange{iv});
              nestedBuilder.create<executor::PrintOp>(
                  loc, ValueRange{elem}, nestedBuilder.getStringAttr("%d "));
              nestedBuilder.create<scf::YieldOp>(loc, ValueRange{});
            });
        moduleBuilder.create<executor::PrintOp>(
            loc, ValueRange{}, moduleBuilder.getStringAttr(""));
      }

      // Verify sorting
      Value numElementsMinusOne =
          moduleBuilder.create<arith::SubIOp>(loc, numElements, c1);
      auto verifyLoop = moduleBuilder.create<scf::ForOp>(
          loc, c0, numElementsMinusOne, c1, ValueRange{c0_i32});
      {
        OpBuilder::InsertionGuard guard(moduleBuilder);
        moduleBuilder.setInsertionPointToStart(verifyLoop.getBody());
        Value i = verifyLoop.getInductionVar();
        Value errors = verifyLoop.getRegionIterArg(0);

        Value curr = moduleBuilder.create<tensor::ExtractOp>(loc, sortedHost,
                                                             ValueRange{i});
        Value iPlusOne = moduleBuilder.create<arith::AddIOp>(loc, i, c1);
        Value next = moduleBuilder.create<tensor::ExtractOp>(
            loc, sortedHost, ValueRange{iPlusOne});
        Value cmp = moduleBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, curr, next);

        auto ifOp = moduleBuilder.create<scf::IfOp>(
            loc, TypeRange{moduleBuilder.getI32Type()}, cmp, true);
        {
          OpBuilder::InsertionGuard guard(moduleBuilder);
          moduleBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          // Count violation without printing details
          Value errorPlusOne =
              moduleBuilder.create<arith::AddIOp>(loc, errors, c1_i32);
          moduleBuilder.create<scf::YieldOp>(loc, ValueRange{errorPlusOne});
        }
        {
          OpBuilder::InsertionGuard guard(moduleBuilder);
          moduleBuilder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          moduleBuilder.create<scf::YieldOp>(loc, ValueRange{errors});
        }

        moduleBuilder.create<scf::YieldOp>(loc, ifOp.getResults());
      }
      Value numErrors = verifyLoop.getResult(0);

      // Print violation count only
      moduleBuilder.create<executor::PrintOp>(
          loc, ValueRange{numErrors},
          moduleBuilder.getStringAttr("Sort violations found: %d"));

      return numErrors;
    }; // End of generateTestForSize lambda

    // Run tests for all sizes
    Value totalErrors = c0_i32;
    for (auto size : testSizes) {
      Value errors = generateTestForSize(size);
      totalErrors =
          moduleBuilder.create<arith::AddIOp>(loc, totalErrors, errors);
    }

    // Return total number of errors across all tests
    moduleBuilder.create<func::ReturnOp>(loc, ValueRange{totalErrors});

    // Print the module in pretty format (not generic form)
    OpPrintingFlags flags;
    flags.printGenericOpForm(false);
    module->print(of->os(), flags);
    of->os() << "\n";

    of->keep();
  }

  std::cout
      << "\nGenerated merge-sort-test-original.mlir in build directory with:\n";
  std::cout << "  - Direct dispatch tests for sizes: ";
  for (size_t i = 0; i < testSizes.size(); i++) {
    std::cout << testSizes[i];
    if (i < testSizes.size() - 1)
      std::cout << ", ";
  }
  std::cout << std::endl;
}
