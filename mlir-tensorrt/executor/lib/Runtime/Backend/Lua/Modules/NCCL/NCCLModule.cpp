//===- NCCLModule.cpp -----------------------------------------------------===//
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
/// Executor NCCL module runtime implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Lua/Modules/NCCL/NcclModule.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include <chrono>

#define OMPI_SKIP_MPICXX
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "mpi.h"
#include "nccl.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

using ExecPtr = uintptr_t;

namespace {

/// A wrapper around a NcclComm_t and some other metadata.
struct NcclCommunicator {
  ncclComm_t comm;
  // Equivalent to ncclCommUserRank.
  int32_t rank;
  // Equivalent to ncclCommCount.
  int32_t numRanks;
};

/// A simple RAII class that checks whether a specified `limit` number of
/// milliseconds have elapsed since the objects creation.
class TimeoutChecker {
public:
  explicit TimeoutChecker(std::chrono::milliseconds limit)
      : start(std::chrono::steady_clock::now()), limit(limit) {}

  /// Check whether the timeout limit has been exceeded and update internal
  /// flag.
  bool operator()() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    exceeded = elapsed > limit;
    return exceeded;
  }

  /// Return true if the timeout was exceeded in the last call.
  bool exceededLimit() const { return exceeded; }

private:
  bool exceeded = false;
  std::chrono::steady_clock::time_point start;
  std::chrono::milliseconds limit;
};
} // namespace

// Wait for the status of `comm` to be `ncclSuccess`. If an async error is
// detected, then returns an error status.
static Status waitUntilNcclCommunicatorIsReady(
    ncclComm_t comm,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  TimeoutChecker checkTimeout(timeout);
  ncclResult_t state;
  do {
    ncclResult_t getLastErrorResult = ncclCommGetAsyncError(comm, &state);
    if (getLastErrorResult != ncclSuccess)
      return getInternalErrorStatus("NCCL failed on ncclCommGetAsyncError");
  } while (state == ncclInProgress && checkTimeout() != true);
  if (checkTimeout.exceededLimit())
    return getInternalErrorStatus("timed out waiting for NCCL operations to "
                                  "compelete");
  if (ncclSuccess != state)
    return getInternalErrorStatus("NCCL experienced an async error: {0}",
                                  ncclGetErrorString(state));
  return getOkStatus();
}

/// Cleans up a specified `NcclCommunicator`.
/// TODO: we should add an error flag to the `NcclCommunicator` struct. If the
/// communicator is in an error state, then we should invoke `ncclCommAbort`
/// instead of the finalization+destruction sequence.
static void destroyNcclCommunicator(uintptr_t ptr) {
  auto *obj = reinterpret_cast<NcclCommunicator *>(ptr);

  if (obj->comm) {
    MTRT_DBGF("Destroying NCCL communicator: %lu",
              reinterpret_cast<uintptr_t>(obj->comm));
    Status waitStatus = waitUntilNcclCommunicatorIsReady(obj->comm);
    if (!waitStatus.isOk()) {
      llvm::errs() << "Error while waiting for NCCL communicator to be ready "
                      "prior to finalizing: "
                   << waitStatus.getString() << "\n";
    }
    ncclResult_t ncclErr;
    ncclErr = ncclCommDestroy(obj->comm);
    if (ncclErr != ncclSuccess && ncclErr != ncclInProgress) {
      llvm::errs() << "ncclCommDestroy error: " << ncclGetErrorString(ncclErr)
                   << "\n";
    }
    obj->comm = nullptr;
  }
  delete obj;
}

namespace {
/// RAII wrapper for NCCL communicator.
struct NcclCommWrapper : public PointerWrapper<NcclCommunicator *> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<NcclCommunicator *> create(ResourceTracker *tracker,
                                             int32_t numRanks,
                                             ncclUniqueId commId,
                                             int32_t rank) {
    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    RETURN_ERROR_IF_NCCL_ERROR(
        ncclCommInitRankConfig(&comm, numRanks, commId, rank, &config),
        nullptr);

    // We cannot split until the source communicator is ready.
    RETURN_STATUS_IF_ERROR(waitUntilNcclCommunicatorIsReady(comm));

    /// TODO: sync abortFlag among all healthy ranks and abort if there is an
    /// error status.

    MTRT_DBGF("Created NCCL communicator: %lu",
              reinterpret_cast<uintptr_t>(comm));
    NcclCommunicator *result = new NcclCommunicator{comm, rank, numRanks};
    tracker->track(reinterpret_cast<uintptr_t>(result),
                   destroyNcclCommunicator);
    return result;
  }

  /// Create a new communicator by splitting an existing one.
  /// TODO: refactoring thandling of `NcclCommunicator` to not require
  /// packaging rank + num_ranks with the communicator would enable
  ///  interleaved work between `ncclCommSplit` and the first use of
  /// the communicator (and where `waitUntilNcclCommunicator` is ready would be
  /// called).
  static StatusOr<NcclCommunicator *> create(ResourceTracker *tracker,
                                             NcclCommunicator *comm,
                                             int32_t color, int32_t key) {
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 18, 1)
    return getStatusWithMsg(
        StatusCode::InternalError,
        "NCCL 2.18.1 or greater is required for ncclCommSplit.");
#else
    // We cannot split until the source communicator is ready.
    RETURN_STATUS_IF_ERROR(waitUntilNcclCommunicatorIsReady(comm->comm));

    // Create a non-blocking communicator with shared resources.
    ncclComm_t newComm = nullptr;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    config.splitShare = 1;
    RETURN_ERROR_IF_NCCL_ERROR(
        ncclCommSplit(comm->comm, color, key, &newComm, &config), comm->comm);
    MTRT_DBGF("Created NCCL communicator via split: %lu",
              reinterpret_cast<uintptr_t>(comm));

    // The communicator is non-blocking, but we use it immediately below, so we
    // must wait until it is ready. For split, this is done by waiting on the
    // source communicator.
    RETURN_STATUS_IF_ERROR(waitUntilNcclCommunicatorIsReady(comm->comm));

    int32_t rank, numRanks;
    RETURN_ERROR_IF_NCCL_ERROR(ncclCommUserRank(newComm, &rank), comm->comm);
    RETURN_ERROR_IF_NCCL_ERROR(ncclCommCount(newComm, &numRanks), comm->comm);
    NcclCommunicator *result = new NcclCommunicator{newComm, rank, numRanks};
    tracker->track(reinterpret_cast<uintptr_t>(result),
                   destroyNcclCommunicator);
    return result;

#endif
  }

  void releaseObj() {}
};

} // namespace

static void registerNcclOps(sol::state_view &lua, ResourceTracker *tracker) {
  //===----------------------------------------------------------------------===//
  // NCCL - Management Ops
  //===----------------------------------------------------------------------===//

  lua["__nccl_comm_init_rank"] = [tracker](sol::this_state state, int32_t rank,
                                           int32_t numRanks,
                                           int32_t device) -> uintptr_t {
    MTRT_DBGF("%s", "Creating NCCL comm");
    sol::state_view lua(state);
    sol::function get_nccl_unique_id = lua["_get_nccl_unique_id"];
    ncclUniqueId id = get_nccl_unique_id(rank);

    SET_LUA_ERROR_AND_RETURN_IF_CUDART_ERROR(cudaSetDevice(device), state, 0);
    StatusOr<NcclCommunicator *> comm =
        NcclCommWrapper::create(tracker, numRanks, id, rank);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(comm, state, 0);
    return reinterpret_cast<uintptr_t>(*comm);
  };

  lua["__nccl_comm_split"] = [tracker](sol::this_state state, uintptr_t comm,
                                       int32_t color,
                                       int32_t key) -> uintptr_t {
    MTRT_DBGF("__nccl_comm_split comm=0x%lx color=%d key=%d", comm, color, key);
    StatusOr<NcclCommunicator *> newComm = NcclCommWrapper::create(
        tracker, reinterpret_cast<NcclCommunicator *>(comm), color, key);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(newComm, state, 0);
    return reinterpret_cast<uintptr_t>(*newComm);
  };

  lua["__nccl_comm_rank"] = [](sol::this_state state, uintptr_t comm) {
    return reinterpret_cast<NcclCommunicator *>(comm)->rank;
  };

  lua["__nccl_comm_num_ranks"] = [](sol::this_state state, uintptr_t comm) {
    return reinterpret_cast<NcclCommunicator *>(comm)->numRanks;
  };

  //===----------------------------------------------------------------------===//
  // NCCL - Collective Ops
  //===----------------------------------------------------------------------===//

#define CALL_FOR_ALL_TYPES(m, arg0, arg1)                                      \
  m(arg0, arg1, i8, ncclInt8) m(arg0, arg1, ui8, ncclUint8)                    \
      m(arg0, arg1, i32, ncclInt32) m(arg0, arg1, ui32, ncclUint32)            \
          m(arg0, arg1, i64, ncclInt64) m(arg0, arg1, ui64, ncclUint64)        \
              m(arg0, arg1, f16, ncclFloat16) m(arg0, arg1, f32, ncclFloat32)  \
                  m(arg0, arg1, f64, ncclFloat64)                              \
                      m(arg0, arg1, bf16, ncclBfloat16)

#define CALL_FOR_ALL_REDOPS_AND_TYPES(m)                                       \
  CALL_FOR_ALL_TYPES(m, sum, ncclSum)                                          \
  CALL_FOR_ALL_TYPES(m, prod, ncclProd)                                        \
  CALL_FOR_ALL_TYPES(m, min, ncclMin)                                          \
  CALL_FOR_ALL_TYPES(m, max, ncclMax)                                          \
  CALL_FOR_ALL_TYPES(m, avg, ncclAvg)

#define DEFINE_NCCL_ALL_REDUCE_METHOD(opsuffix, op, typesuffix, type)          \
  lua["__nccl_all_reduce_" #opsuffix "_" #typesuffix] =                        \
      [](sol::this_state state, ExecPtr sendbuff, ExecPtr recvbuff,            \
         size_t count, uintptr_t communicator, CudaStreamPtr stream) {         \
        auto comm = reinterpret_cast<NcclCommunicator *>(communicator);        \
        SET_LUA_ERROR_IF_NCCL_ERROR(                                           \
            ncclAllReduce(reinterpret_cast<void *>(sendbuff),                  \
                          reinterpret_cast<void *>(recvbuff), count, type, op, \
                          comm->comm, stream),                                 \
            state, comm->comm);                                                \
        SET_LUA_ERROR_AND_RETURN_IF_ERROR(                                     \
            waitUntilNcclCommunicatorIsReady(comm->comm), state, );            \
      };

  CALL_FOR_ALL_REDOPS_AND_TYPES(DEFINE_NCCL_ALL_REDUCE_METHOD)
#undef DEFINE_NCCL_ALL_REDUCE_METHOD

#define DEFINE_NCCL_REDUCE_SCATTER_METHOD(opsuffix, op, typesuffix, type)      \
  lua["__nccl_reduce_scatter_" #opsuffix "_" #typesuffix] =                    \
      [](sol::this_state state, ExecPtr sendbuff, ExecPtr recvbuff,            \
         size_t recvcount, uintptr_t communicator, CudaStreamPtr stream) {     \
        auto *comm = reinterpret_cast<NcclCommunicator *>(communicator);       \
        SET_LUA_ERROR_IF_NCCL_ERROR(                                           \
            ncclReduceScatter(reinterpret_cast<void *>(sendbuff),              \
                              reinterpret_cast<void *>(recvbuff), recvcount,   \
                              type, op, comm->comm, stream),                   \
            state, comm->comm);                                                \
        SET_LUA_ERROR_AND_RETURN_IF_ERROR(                                     \
            waitUntilNcclCommunicatorIsReady(comm->comm), state, );            \
      };

  CALL_FOR_ALL_REDOPS_AND_TYPES(DEFINE_NCCL_REDUCE_SCATTER_METHOD)
#undef DEFINE_NCCL_REDUCE_SCATTER_METHOD

  lua["__nccl_all_gather"] = [](sol::this_state state, ExecPtr sendbuff,
                                ExecPtr recvbuff, size_t sendNumBytes,
                                uintptr_t communicator, CudaStreamPtr stream) {
    auto *comm = reinterpret_cast<NcclCommunicator *>(communicator);
    SET_LUA_ERROR_IF_NCCL_ERROR(
        ncclAllGather(reinterpret_cast<void *>(sendbuff),
                      reinterpret_cast<void *>(recvbuff), sendNumBytes,
                      ncclInt8, comm->comm, stream),
        state, comm->comm);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(
        waitUntilNcclCommunicatorIsReady(comm->comm), state, );
  };

  lua["__nccl_all_to_all"] = [](sol::this_state state, ExecPtr sendbuff,
                                ExecPtr recvbuff, size_t numBytes,
                                uintptr_t communicator, CudaStreamPtr stream) {
    auto *comm = reinterpret_cast<NcclCommunicator *>(communicator);
    size_t sendBytes = numBytes / comm->numRanks;
    SET_LUA_ERROR_IF_NCCL_ERROR(ncclGroupStart(), state, comm->comm);
    for (int r = 0; r < comm->numRanks; ++r) {
      SET_LUA_ERROR_IF_NCCL_ERROR(
          ncclSend(reinterpret_cast<void *>(sendbuff + r * sendBytes),
                   sendBytes, ncclInt8, r, comm->comm, stream),
          state, comm->comm);
      SET_LUA_ERROR_IF_NCCL_ERROR(
          ncclRecv(reinterpret_cast<void *>(recvbuff + r * sendBytes),
                   sendBytes, ncclInt8, r, comm->comm, stream),
          state, comm->comm);
    }
    SET_LUA_ERROR_IF_NCCL_ERROR(ncclGroupEnd(), state, comm->comm);

    SET_LUA_ERROR_AND_RETURN_IF_ERROR(
        waitUntilNcclCommunicatorIsReady(comm->comm), state, );
  };

  lua["__nccl_permute"] = [](sol::this_state state, ExecPtr sendbuff,
                             ExecPtr recvbuff, int32_t sendId, int32_t recvId,
                             size_t numBytes, uintptr_t communicator,
                             CudaStreamPtr stream) {
    auto *comm = reinterpret_cast<NcclCommunicator *>(communicator);
    if (recvId == -1) {
      // Zero out recvbuff if not receiving.
      SET_LUA_ERROR_IF_CUDA_ERROR(
          cuMemsetD8Async(static_cast<CUdeviceptr>(recvbuff), 0, numBytes,
                          stream),
          state);
    }
    SET_LUA_ERROR_IF_NCCL_ERROR(ncclGroupStart(), state, comm->comm);
    if (sendId != -1) {
      SET_LUA_ERROR_IF_NCCL_ERROR(ncclSend(reinterpret_cast<void *>(sendbuff),
                                           numBytes, ncclInt8, sendId,
                                           comm->comm, stream),
                                  state, comm->comm);
    }
    if (recvId != -1) {
      SET_LUA_ERROR_IF_NCCL_ERROR(ncclRecv(reinterpret_cast<void *>(recvbuff),
                                           numBytes, ncclInt8, recvId,
                                           comm->comm, stream),
                                  state, comm->comm);
    }
    SET_LUA_ERROR_IF_NCCL_ERROR(ncclGroupEnd(), state, comm->comm);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(
        waitUntilNcclCommunicatorIsReady(comm->comm), state, );
  };
}

void mlirtrt::runtime::registerExecutorNCCLModuleLuaRuntimeMethods(
    lua_State *state, ResourceTracker *tracker) {
  sol::state_view lua(state);
  registerNcclOps(lua, tracker);
}

StatusOr<std::string> mlirtrt::runtime::getCommunicatorUniqueId() {
  ncclUniqueId id;
  RETURN_ERROR_IF_NCCL_ERROR(ncclGetUniqueId(&id), nullptr);
  std::string asString = std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
  MTRT_DBGF("NCCL unique id: %s", asString.c_str());
  return asString;
}

void mlirtrt::runtime::registerDeviceDependentNCCLMethods(
    lua_State *state, int32_t numDevices, int32_t deviceIdx,
    llvm::StringRef ncclUuid) {
  sol::state_view lua(state);
  lua["__spmd_global_num_ranks"] = [numDevices](sol::this_state state) {
    return numDevices;
  };
  lua["__spmd_global_rank"] = [deviceIdx](sol::this_state state) {
    return deviceIdx;
  };

  ncclUniqueId id;
  std::copy_n(ncclUuid.begin(), ncclUuid.size(), id.internal);
  lua["_get_nccl_unique_id"] = [id](sol::this_state state, int32_t rank) {
    return id;
  };
}
