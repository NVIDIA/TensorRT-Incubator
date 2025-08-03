//===- TensorKindOpInterface.td ----------------------------*- Tablegen -*-===//
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
#ifndef MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_TD
#define MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def TensorKindOpInterface : OpInterface<"TensorKindOpInterface"> {
  let description = [{
    The 'TensorKindOpInterface' is implemented by operations to participate
    in the TensorKindAnalysis. The interface lets operations descibe the
    "backwards" transfer function for determining the tensor kinds of the
    operands given the tensor kinds of the results.

    See also the
    [`TensorKindAnalysis` documentation](docs/Analysis/TensorKindAnalysis.md)
    for more information.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Performs the tensor kind transfer function",
      /*retTy=*/"void",
      /*methodName=*/"inferOperandKind",
      /*args=*/(ins "ArrayRef<TensorKindLattice*>":$operands,
                    "ArrayRef<const TensorKindLattice*>":$results,
                    "llvm::function_ref<void(OpOperand &operand, TensorKind kind)>":$setInferredType),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/[{
        Return TensorKind::[Host|Device] if OpOperand should be placed in host
        or device memory during bufferization (regardless of the space of result
        operands). Otherwise, return TensorKind::Unkown or TensorKind::Both (any
        value other than Host or Device indicates the kind depends on dataflow
        analysis.
      }],
      /*retTy=*/"::mlir::TensorKind",
      /*methodName=*/"getStaticOperandTensorKind",
      /*args=*/(ins "::mlir::OpOperand&":$operand),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    StaticInterfaceMethod<
      /*desc=*/[{
        Returns true if the given value is a candidate for a host tensor based on
        its type information. It must be a statically-shaped integer tensor with
        fewer than 8 elements.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isHostTensorCandidate",
      /*args=*/(ins "::mlir::Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::isHostTensorCandidate(value.getType());
      }]>
  ];
}


#endif // MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_TD
