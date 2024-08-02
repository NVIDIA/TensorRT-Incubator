//===- MPI.cpp ------------------------------------------------------------===//
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
/// Definitions of utilities for setting up and tearing down MPI.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Support/MPI.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "llvm/Support/raw_ostream.h"

#ifdef MLIR_TRT_ENABLE_NCCL
#define OMPI_SKIP_MPICXX
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "mpi.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

static Status getMPIErrorStatus(llvm::StringRef msg, int32_t errCode) {
  llvm::SmallString<MPI_MAX_ERROR_STRING> str;
  str.resize(MPI_MAX_ERROR_STRING);
  int errClass = 0;
  int errStrLen = 0;
  MPI_Error_class(errCode, &errClass);
  MPI_Error_string(errCode, str.data(), &errStrLen);
  str.resize(errStrLen);
  return getInternalErrorStatus("{0}: [class={1}, msg={2}]", msg, errClass,
                                str);
}

StatusOr<std::unique_ptr<MPIManager>> MPIManager::create() {
  int status = MPI_Init(nullptr, nullptr);
  if (status != MPI_SUCCESS)
    return getMPIErrorStatus("MPI_init failed", status);

  MTRT_DBGF("%s", "MPI_Init succeeded");
  return std::unique_ptr<MPIManager>(new MPIManager());
}

MPIManager::MPIManager() {}

MPIManager::~MPIManager() {
  MTRT_DBGF("%s", "Calling MPI_Finalize");
  int status = MPI_Finalize();
  if (status != MPI_SUCCESS) {
    Status result = getMPIErrorStatus("MPI_Finalize failed", status);
    llvm::errs() << result.getString() << "\n";
  }
}

#endif //  MLIR_TRT_ENABLE_NCCL
