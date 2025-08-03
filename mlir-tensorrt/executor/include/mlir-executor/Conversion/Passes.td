//===- Passes.td -------------------------------------------*- Tablegen -*-===//
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
#ifndef INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES
#define INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

defvar ConvertToExecutorOptions = [
  Option<"indexBitwidth", "index-bitwidth", "int64_t", "64",
  "all index types will be converted to signless integers of this bitwidth">
];

def ConvertLinalgToExecutorPass : Pass<"convert-linalg-to-executor"> {
  let summary = "Convert Linalg upstream dialect ops to 'executor' dialect";
  let description = [{
    The purpose of this conversion is to lower certain Linalg operations to
    the Executor dialect.

    Only certain bufferized Linalg operations are supported:

    - 'linalg.fill' filling a device buffer where the element type has size of
      1, 2, or 4 bytes.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
  let options = ConvertToExecutorOptions;
}

def ConvertMemRefToExecutorPass : Pass<"convert-memref-to-executor"> {
  let summary = "Convert MemRef dialect ops to Exectuor dialect ops";
  let description = [{
    The purpose of this conversion is to lower `memref` operations to more
    primitive operations of the Executor dialect. After this pass, the any
    `memref` operations should be converted and the `memref` type should be
    eliminated everywhere except at function boundaries. Remaining uses of the
    `memref` type as function arguments and at call sites are eliminated by the
    func-to-executor patterns applied in the `std-to-executor` pass.

  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect",
    ];
  let options = !listconcat(ConvertToExecutorOptions, [
    Option<"allowUncheckedMemrefCastConversion", "allow-unchecked-memref-cast-conversion",
        "bool", "true", "allow converting memref casts without runtime checks">
  ]);
}

def ConvertStdToExecutorPass : Pass<"convert-std-to-executor"> {
  let summary = "Convert standard upstream dialect ops to 'executor' dialect";
  let description = [{
    The purpose of this conversion is to lower or modify IR containing
    operations from three dialects (ControlFlow, Arith, Func) to prepare for
    translation to an interpreter or conversion to another lower-level dialect.

    This pass has the following prerequisuites:

    1. Any Executor op that should be lowered to a `func.call` must already have
       been lowered.

    2. No `memref` dialect operations should be present. Any usage of the
       `memref` type will be expanded to Executor dialect `table` types
       representing descriptors.

    3. There should be no `scf` dialect operations present. The
       `convert-scf-to-cf` should be used to lower `scf` operations.

    After this pass runs:

    1. No index types should be present. Index types are converted to signless
       integers of the specified bitwidth.

    2. No `memref` types should be present. All memref types, including at
       function boundaries, should be converted to `!executor.table<>` types
       that represent descriptors.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
    "::mlir::math::MathDialect"
  ];
  let options = ConvertToExecutorOptions;
}

def ConvertExecutorToExecutorPass : Pass<"convert-executor-to-executor"> {
  let summary = "Perform exector-to-executor lowerings";
  let description = [{
    The purpose of this conversion is to lower `executor` operations that
    operate on high-level types like `memref` or `index` into lower level
    types. These are mostly structural conversions, such as performing
    type conversions for the `executor.func` and `executor.call` operations.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
  let options = ConvertToExecutorOptions;
}

#endif // INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES
