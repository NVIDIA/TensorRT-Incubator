//===- TensorKindAttrInterface.td --------------------------*- Tablegen -*-===//
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
#ifndef MLIR_TENSORRT_INTERFACE_TENSORKINDATTRINTERFACE_TD
#define MLIR_TENSORRT_INTERFACE_TENSORKINDATTRINTERFACE_TD

include "mlir/IR/OpBase.td"

def TensorKindAttrInterface : AttrInterface<"TensorKindAttrInterface"> {

  let description = [{

    The 'TensorKindAttrInterface' is implemented by attributes that are
    able to determine whether a tensor type should be interpreted as
    belonging to a certain memory space (e.g. 'device', 'host', 'both', or
    'unknown').

    This interface should be implemented by attributes that are used
    as tensor type encoding attributes or used as a memory space in
    bufferization ops like `bufferization.alloc_tensor'.

  }];

  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Performs the tensor kind transfer function",
      /*retTy=*/"TensorKindInfo",
      /*methodName=*/"getTensorKind",
      /*args=*/(ins ),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
  ];
}

#endif // MLIR_TENSORRT_INTERFACE_TENSORKINDATTRINTERFACE_TD
