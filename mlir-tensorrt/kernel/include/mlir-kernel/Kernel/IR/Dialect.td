//===- Dialect.td ---------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELDIALECT_TD
#define MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Kernel Dialect
//===----------------------------------------------------------------------===//

def Kernel_Dialect : Dialect {
  let name = "kernel";
  let summary = "A dialect that provides ops and transforms for codegen";
  let description = [{
    The kernel dialect provides the following:

      1. Operations that fill some requirement not currently provided
         by an upstream op during the process of lowering "linalg on tensors".

      2. A central place to organize additional transforms/passes and
         Transform IR extension operations that are not provided by
         upstream MLIR.
  }];
  let cppNamespace = "::mlir::kernel";

  // Enable when the dialect gains custom types.
  // let useDefaultTypePrinterParser = 1;

  let useDefaultAttributePrinterParser = 1;

  // KernelDialect has region argument attributes that need to be verified,
  // for example, `kernel.alignment`.
  let hasRegionArgAttrVerify = 1;

  // KernelDialect has discardable attributes that need to be verified,
  // for example, `kernel.num_threads` attached to `func.func`.
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Return the name of the attribute that is attached to a
    /// `gpu.module` and holds compiled PTX data.
    static constexpr StringRef getKernelModulePtxDataAttrName() {
      return "kernel.ptx_data";
    }

    /// Returns the name of the attribute that specifies the GPU module
    /// kind.
    static constexpr StringRef getGpuModuleKindAttrName() {
      return "kernel.gpu_module_kind";
    }

    /// The argument attribute key which describes the expected alignment of an input buffer.
    /// Only valid for tensor types and is removed after bufferization and materialization
    /// of `memref.assume_alignment` ops.
    static constexpr llvm::StringLiteral kKernelAlignmentArgAttrName = "kernel.alignment";

    /// The argument attribute key which describes the expected alignment size
    /// of bufferization.alloc_tensor.
    static constexpr llvm::StringLiteral kBufferlizationAllocationAlignmentAttrName = "kernel.alignment";

    /// Return a const reference to the transform schedule registry.
    const TransformScheduleRegistry &getTransformScheduleRegistry() const {
      return transformScheduleGeneratorRegistry;
    }

    /// Declare the registration implementation as a friend so that it can
    /// modify the private member `transformScheduleGeneratorRegistry`.
    friend void ::mlir::kernel::detail::registerTransformScheduleGenerator(
      KernelDialect *, std::unique_ptr<TransformScheduleBase> );


    /// Returns the name of the i64 integer attribute that describes how many
    /// threads should be allocated to a threadblock for a particular kernel.
    /// This attribute should be attached to `func.func` nested in `gpu.module`
    /// operations.
    static constexpr StringRef getKernelFunctionNumThreadsAttrName() {
      return "kernel.num_threads";
    }

    /// Returns the number of threads required for a particular kernel.
    /// This function should be called on `func.func` nested in `gpu.module`
    /// operations.
    static std::optional<int64_t> getNumThreadsRequired(FunctionOpInterface func);


    /// Returns the name of the attribute that specifies the warp size for a
    /// particular GPU module.
    /// TODO: this should use DLTI instead
    static constexpr StringRef getGpuModuleWarpSizeAttrName() {
      return "kernel.warp_size";
    }

    /// Returns the warp size for a particular GPU module. Generally, this is
    /// 32 (default for all current NV architectures), but in the future if the number
    /// changes we can override on a per-module basis.
    static int64_t getGpuModuleWarpSize(gpu::GPUModuleOp gpuModule);

private:
    /// A registry containing all the available transform schedule generators.
    TransformScheduleRegistry transformScheduleGeneratorRegistry;

    void registerAttributes();
    void registerOps();
  }];

  let dependentDialects = [
    "::mlir::NVVM::NVVMDialect"
  ];
}

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
        Op<Kernel_Dialect, mnemonic, traits>;

#endif // MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELDIALECT_TD
