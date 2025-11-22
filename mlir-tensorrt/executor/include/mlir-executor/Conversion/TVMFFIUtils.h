//===- TVMFFIUtils.h --------------------------------------------*- C++ -*-===//
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
/// This module implements code generation for the TVM FFI ABI calling
/// convention. The TVM FFI ABI uses a unified calling convention where all
/// function arguments and return values are represented as `TVMFFIAny`
/// structures. Each `TVMFFIAny` is a 16-byte tagged union containing:
/// - `type_index` (i32): Identifies the type of the value
/// - `zero_padding` (i32): Padding or metadata (e.g., string length for small
/// strings)
/// - `value` (i64): The actual value (POD types) or pointer (object types)
///
/// POD values (integers, floats) are stored directly in the value field, while
/// heap-allocated objects (like DLTensor) are stored as pointers. DLTensor
/// structures follow the DLPack format and are compatible with TVM FFI's tensor
/// representation.
///
/// The generated IR creates `!executor.table` values that represent `TVMFFIAny`
/// structures in memory, which are then passed to TVM FFI functions following
/// the standard calling convention: `TVMFFISafeCallType(handle, args, num_args,
/// result)`.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-tensorrt-common/Support/Status.h"

namespace mlir {

class Type;
class OpBuilder;
class MLIRContext;

namespace executor {

class MemRefDescriptor;

struct TVMFFIArgsCallHelper {

  TVMFFIArgsCallHelper(OpBuilder &builder, Type indexType);
  ~TVMFFIArgsCallHelper();

  /// Constructs an `!executor.table` value representing a `DLTensor` struct
  /// from the given memref descriptor.
  ///
  /// Creates a DLPack-compatible tensor structure that can be passed to TVM FFI
  /// functions. The DLTensor structure contains:
  /// - Data pointer and device information
  /// - Rank, shape, and strides arrays
  /// - Data type information (DLDataType)
  /// - Byte offset for non-contiguous views
  ///
  /// The returned table can be promoted to an alloca and used as a pointer
  /// in a `TVMFFIAny` with `type_index = kTVMFFIDLTensorPtr`.
  mtrt::StatusOr<Value> createDLTensor(Location loc, Value deviceId,
                                       executor::MemRefDescriptor desc) const;

  /// Promotes a value to an alloca in host memory, returning a pointer to it.
  Value promoteToAlloca(Location loc, Value value) const;

  /// Creates a TVM FFI argument array from the decode specification.
  ///
  /// Converts original and converted arguments/outputs to TVM FFI `Any` values
  /// and arranges them according to the decode spec. This implements the TVM
  /// FFI calling convention where all arguments are packed into an array of
  /// `TVMFFIAny` structures.
  ///
  /// The decode spec determines the ordering and source of each argument:
  /// - `DecodeArg`: Maps to an input argument
  /// - `DecodeRet`: Maps to an output argument (for in-place operations)
  /// - `DecodeAttr`: Maps to an immediate attribute value
  /// - `OptionalNoneTag`: Represents an optional None value
  ///
  /// Returns a table containing the packed `TVMFFIAny` values ready to be
  /// passed to a TVM FFI function following the `TVMFFISafeCallType` signature.
  mtrt::StatusOr<Value> createTVMFFIAnyArrayForPluginCall(
      Location loc, const abi::plugin::DecodeSpec &decodeSpec,
      ValueRange originalArgs, ValueRange convertedArgs,
      ValueRange originalOutputs, ValueRange convertedOutputs,
      DictionaryAttr attrDict, Value deviceID) const;

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace executor
} // namespace mlir
