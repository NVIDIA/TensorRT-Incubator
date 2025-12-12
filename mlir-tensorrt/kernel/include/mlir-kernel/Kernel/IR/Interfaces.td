//===- Interfaces.td ------------------------------------------------------===//
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

#ifndef MLIR_KERNEL_KERNEL_IR_INTERFACES
#define MLIR_KERNEL_KERNEL_IR_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CodegenScheduleAttrInterface
//===----------------------------------------------------------------------===//

def CodegenScheduleAttrInterface : AttrInterface<"CodegenScheduleAttrInterface"> {
  let cppNamespace = "::mlir::kernel";

  let description = [{

    The `CodegenScheduleAttrInterface` is implemented by codegen schedule attributes
    that describe what is the schedule parameters for certain operations.

  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the CTA tile shape. One integer for every iterator of the
        linalg operation.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getCTATileShape",
      /*args=*/(ins "linalg::LinalgOp":$op),
      /*body=*/"",
      /*defaultBody=*/""
    >
  ];
}

//===----------------------------------------------------------------------===//
// GPUModuleLoweringAttrInterface
//===----------------------------------------------------------------------===//

def GPUModuleLoweringAttrInterface
      : AttrInterface<"GPUModuleLoweringAttrInterface"> {
  let cppNamespace = "::mlir::kernel";

  let description = [{
    The `GPUModuleLoweringAttrInterface` is implemented by attributes that
    describe how a `gpu.module` should be lowered to the final target binary.

    The Kernel Dialect's kernel generation pipelines (especially the
    Transform IR pipeline) may produce different `gpu.module` which have
    distinct lowering requirements. This attribute is a way of communicating,
    in an extensible manner, how to perform the final lowering of the
    `gpu.module` IR.

    The TransformSchedule infrastructure is open (projects external to the
    Kernel dialect can declare and register new schedules), and therefore
    they should create their own attributes that implement this interface
    unless the existing Kernel GPUModuleLowering attributes are sufficient.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Adds pases for processing the `gpu.module` IR. for the
        specified compilation phase.
      }],
      /*retTy=*/"std::optional<LogicalResult>",
      /*methodName=*/"addPhasePasses",
      /*args=*/(ins "mlir::gpu::GPUModuleOp":$gpu_module,
                    "OpPassManager&":$gpu_module_pass_manager,
                    "GPUModuleLoweringPhase":$phase,
                    "StringRef":$debugDumpDir),
      /*body=*/"",
      /*defaultBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the default layout map for tensor or memref arguments
        of kernel entrypoint functions in this module.
      }],
      /*retTy=*/"KernelArgLayoutMapOption",
      /*methodName=*/"getMemRefArgumentDefaultLayoutMap",
      /*args=*/(ins "mlir::gpu::GPUModuleOp":$gpu_module),
      /*body=*/"",
      /*defaultBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether this GPU module kind should participate in
        module bufferization. If false, no operations within the module
        will be bufferized and post-bufferization actions will be skipped.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"shouldParticipateInModuleBufferization",
      /*args=*/(ins "mlir::gpu::GPUModuleOp":$gpu_module),
      /*body=*/"",
      /*defaultBody=*/"return true;"
    >
  ];
}

//===----------------------------------------------------------------------===//
// PointerLikeTypeInterface
//===----------------------------------------------------------------------===//

def PointerLikeTypeInterface : TypeInterface<"PointerLikeTypeInterface"> {
  let cppNamespace = "::mlir::kernel";
  let description = [{
    A lightweight interface to identify pointer-like types and optionally query
    their pointee/element type. Implementations may return a null Type if the
    pointee is not representable (e.g., opaque pointers).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the pointee/element type or a null Type if opaque.",
      /*retTy=*/"Type",
      /*methodName=*/"getPointeeType",
      /*args=*/(ins),
      /*body=*/"",
      /*defaultImpl=*/"return nullptr;"

    >
  ];
}

def GPUModuleLoweringAttr :
     ConfinedAttr<AnyAttr, [PromisedAttrInterface<GPUModuleLoweringAttrInterface>]> {
  let description = [{
    Generic GPU module lowering attribute. These attributes must implement or promise
    the `GPUModuleLoweringAttrInterface` interface.
  }];
}

#endif // MLIR_KERNEL_KERNEL_IR_INTERFACES
