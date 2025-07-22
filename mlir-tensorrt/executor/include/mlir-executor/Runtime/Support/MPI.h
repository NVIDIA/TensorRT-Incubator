//===- MPI.h ----------------------------------------------------*- C++ -*-===//
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
///
/// Declarations for MPI runtime utilities.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_SUPPORT_MPI
#define MLIR_EXECUTOR_RUNTIME_SUPPORT_MPI

#include "mlir-tensorrt-common/Support/Status.h"
#include <memory>

namespace mlirtrt::runtime {

/// The MPIManager calls MPI_Init on creation and MPI_Finalize upon destruction.
/// The downstream applications that are potentially executing executables using
/// MPI (e.g. mlir-tensorrt-runner) should setup the manager once per process
/// close to the time when the runtime is initialized.
class MPIManager {
public:
  /// Creates an instance of MPIManager and calls MPI_Init. Returns the MPI
  /// error if it occurs.
  static StatusOr<std::unique_ptr<MPIManager>> create();

  /// Calls MPI_Finalize.
  ~MPIManager();

private:
  MPIManager();
};

} // namespace mlirtrt::runtime

#endif // MLIR_EXECUTOR_RUNTIME_SUPPORT_MPI
