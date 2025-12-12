//===- Enums.td -----------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELENUMS_TD
#define MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELENUMS_TD

include "mlir-kernel/Kernel/IR/Dialect.td"

include "mlir/IR/EnumAttr.td"

class Kernel_I32EnumAttr<string name, string description,
  list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::kernel";
  let genSpecializedAttr = 0;
}

class Kernel_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic> :
  EnumAttr<Kernel_Dialect, intEnumAttr, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Kernel_GPUModuleLoweringPhase : Kernel_I32EnumAttr<
  "GPUModuleLoweringPhase",
  "phases of GPU module lowering",
  [
    I32EnumAttrCase<"PreBufferization", 0, "pre_bufferization">,
    I32EnumAttrCase<"PostBufferization", 1, "post_bufferization">,
    I32EnumAttrCase<"LowerToNVVM", 2, "lower_to_nvvm">
  ]
>;
def Kernel_GPUModuleLoweringPhaseAttr : Kernel_EnumAttr<
  Kernel_GPUModuleLoweringPhase,
  "gpu_module_lowering_phase">;

def Kernel_KernelArgLayoutMapOption : Kernel_I32EnumAttr<
  "KernelArgLayoutMapOption",
  "options for kernel arg layout map",
  [
    I32EnumAttrCase<"Identity", 0, "identity">,
    I32EnumAttrCase<"FullyDynamic", 1, "fully_dynamic">
  ]
>;

#endif // MLIR_TENSORRT_DIALECT_KERNEL_IR_KERNELENUMS_TD
