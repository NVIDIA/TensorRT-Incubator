//===- CuBLASModule.cpp ---------------------------------------------------===//
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
/// Executor CuBLAS module runtime implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/Backend/Lua/SolAdaptor.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#endif
#include "cublasLt.h"
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace mtrt;
using namespace mtrt;

#define SET_LUA_ERROR_IF_CUBLAS_ERROR(x, lstate, msg)                          \
  do {                                                                         \
    cublasStatus_t err = (x);                                                  \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::stringstream ss;                                                    \
      lua_State *L = lstate;                                                   \
      ss << __FILE__ << ":" << __LINE__ << " " << msg << ": " << err;          \
      luaL_error(L, ss.str().c_str());                                         \
    }                                                                          \
  } while (false)

#define RETURN_ERROR_IF_CUBLAS_ERROR(x, msg)                                   \
  do {                                                                         \
    cublasStatus_t err = (x);                                                  \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      return getInternalErrorStatus("{0}:{1} {2}; {3} {4}", __FILE__,          \
                                    __LINE__, msg, cublasGetStatusName(err),   \
                                    cublasGetStatusString(err));               \
    }                                                                          \
  } while (false);

//===----------------------------------------------------------------------===//
// RAII wrappers for `cublasLtHandle_t`
//===----------------------------------------------------------------------===//

namespace {
struct CublasLtHandle : public PointerWrapper<cublasLtHandle_t> {
  using PointerWrapper::PointerWrapper;

  static StatusOr<CublasLtHandle> create(ResourceTracker &tracker) {
    cublasLtHandle_t handle;
    RETURN_ERROR_IF_CUBLAS_ERROR(cublasLtCreate(&handle),
                                 "could not create cuBLAS handle");
    MTRT_DBGF("created cuBLAS handle 0x%lx",
              reinterpret_cast<uintptr_t>(handle));
    CublasLtHandle result(handle);
    tracker.track(result, [](uintptr_t ptr) {
      if (ptr) {
        CublasLtHandle wrapper(ptr);
        MTRT_DBGF("Destroying cuBLAS Lt handle: 0x%lx", ptr);
        cublasLtDestroy(wrapper);
      }
    });
    return result;
  };
};
} // namespace

//===----------------------------------------------------------------------===//
// RAII wrappers for `cublasLtMatmulHeuristicResult_t`
//===----------------------------------------------------------------------===//

static std::tuple<cublasComputeType_t, cudaDataType_t>
getCublasComputeAndDataType(mtrt::ScalarTypeCode scalarType) {
  switch (scalarType) {
  case (mtrt::ScalarTypeCode::f16):
    return std::make_tuple(cublasComputeType_t::CUBLAS_COMPUTE_16F,
                           cudaDataType_t::CUDA_R_16F);
  case (mtrt::ScalarTypeCode::f32):
    return std::make_tuple(cublasComputeType_t::CUBLAS_COMPUTE_32F,
                           cudaDataType_t::CUDA_R_32F);
  case (mtrt::ScalarTypeCode::f64):
    return std::make_tuple(cublasComputeType_t::CUBLAS_COMPUTE_64F,
                           cudaDataType_t::CUDA_R_64F);
  case (mtrt::ScalarTypeCode::i32):
    return std::make_tuple(cublasComputeType_t::CUBLAS_COMPUTE_32I,
                           cudaDataType_t::CUDA_R_32I);
  default:
    llvm_unreachable("unhandled or invalid data type to convert to cuBLAS");
  }
}

// cublasLtMatmulTile_t is an enumerated type used to set the tile size as MxN.
// The mapping here is based on:
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmultile-t

const static std::vector<std::pair<int, int>> matmulTileIds = {
    {0, 0}, // tile size is undefined in this senario
    {8, 8},     {8, 16},    {16, 8},   {8, 32},   {16, 16},   {32, 8},
    {8, 64},    {16, 32},   {32, 16},  {64, 8},   {32, 32},   {32, 64},
    {64, 32},   {32, 128},  {64, 64},  {128, 32}, {64, 128},  {128, 64},
    {64, 256},  {128, 128}, {256, 64}, {64, 512}, {128, 256}, {256, 128},
    {512, 64},  {64, 96},   {96, 64},  {96, 128}, {128, 160}, {160, 128},
    {192, 128}, {128, 192}, {128, 96}};

static double euclideanDistance(int x1, int y1, int x2, int y2) {
  return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// This class holds `cublasLtMatmulPreference_t`, `cublasLtMatrixLayout_t`,
// `cublasLtMatmulDesc_t`, and actual algorithm selection result
// `cublasLtMatmulHeuristicResult_t`. These parameters are hold together since
// all of them are specific to a problem size and are reused if similar problem
// size is encountered in the graph later.
namespace {
class AlgoSelectResult {
public:
  static StatusOr<std::unique_ptr<AlgoSelectResult>>
  get(CublasLtHandle handle, llvm::ArrayRef<int64_t> algoSelectorArgs) {
    auto r = std::make_unique<AlgoSelectResult>();
    MTRT_RETURN_IF_ERROR(r->prepare(algoSelectorArgs));
    MTRT_RETURN_IF_ERROR(r->selectAlgo(handle));
    return r;
  };
  cublasLtMatmulPreference_t &getMatmulPreference() { return preference; }
  cublasLtMatmulDesc_t &getMatmulDesc() { return operationDesc; }
  cublasLtMatrixLayout_t &getALayout() { return aLayout; }
  cublasLtMatrixLayout_t &getBLayout() { return bLayout; }
  cublasLtMatrixLayout_t &getCLayout() { return cLayout; }
  cublasLtMatmulHeuristicResult_t &getAlgo() { return algo; }
  ~AlgoSelectResult() {
    if (preference) {
      MTRT_DBGF("Destroying MatMul preference handle: %lu",
                reinterpret_cast<uintptr_t>(preference));
      cublasLtMatmulPreferenceDestroy(preference);
      preference = nullptr;
    }
    if (aLayout) {
      MTRT_DBGF("Destroying Mat A layout handle: %lu",
                reinterpret_cast<uintptr_t>(aLayout));
      cublasLtMatrixLayoutDestroy(aLayout);
      aLayout = nullptr;
    }
    if (bLayout) {
      MTRT_DBGF("Destroying Mat B layout handle: %lu",
                reinterpret_cast<uintptr_t>(bLayout));
      cublasLtMatrixLayoutDestroy(bLayout);
      bLayout = nullptr;
    }
    if (cLayout) {
      MTRT_DBGF("Destroying Mat C layout handle: %lu",
                reinterpret_cast<uintptr_t>(cLayout));
      cublasLtMatrixLayoutDestroy(cLayout);
      cLayout = nullptr;
    }
    if (operationDesc) {
      MTRT_DBGF("Destroying MatMul description handle: %lu",
                reinterpret_cast<uintptr_t>(operationDesc));
      cublasLtMatmulDescDestroy(operationDesc);
      operationDesc = nullptr;
    }
  }

private:
  Status prepare(llvm::ArrayRef<int64_t> algoSelectorArgs) {
    assert(algoSelectorArgs.size() == 19 &&
           "expected 19 arguments to cuBLAS algorithm selection");
    MTRT_DBGF("%s", "[cuBLAS] preparing for matmul and algorithm selection");
    // First element of the argument tells about data type
    std::tuple computeAndDataType = getCublasComputeAndDataType(
        static_cast<mtrt::ScalarTypeCode>(algoSelectorArgs[0]));
    MTRT_DBGF("cublas type: %d, cuda type: %d", std::get<0>(computeAndDataType),
              std::get<1>(computeAndDataType));
    int32_t batchSize = algoSelectorArgs[1];
    MTRT_DBGF("batch size: %d", batchSize);
    // Get matrix operation information
    auto transposeA = static_cast<cublasOperation_t>(algoSelectorArgs[6]);
    auto transposeB = static_cast<cublasOperation_t>(algoSelectorArgs[11]);
    auto transposeC = static_cast<cublasOperation_t>(algoSelectorArgs[16]);
    // Create matmul descriptor and set attributes
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulDescCreate(&operationDesc,
                                 std::get<0>(computeAndDataType),
                                 std::get<1>(computeAndDataType)),
        "could not create cuBLAS matmul descriptor");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulDescSetAttribute(operationDesc,
                                       CUBLASLT_MATMUL_DESC_TRANSA, &transposeA,
                                       sizeof(transposeA)),
        "failed to set cuBLAS matmul descriptor attribute");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulDescSetAttribute(operationDesc,
                                       CUBLASLT_MATMUL_DESC_TRANSB, &transposeB,
                                       sizeof(transposeB)),
        "failed to set cuBLAS matmul descriptor attribute");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulDescSetAttribute(operationDesc,
                                       CUBLASLT_MATMUL_DESC_TRANSC, &transposeC,
                                       sizeof(transposeC)),
        "failed to set cuBLAS matmul descriptor attribute");
    MTRT_DBGF("%s", "[cuBLAS] created matmul descriptor and set attributes");
    // Create layout for each input and output matrix, and set attributes
    cublasLtOrder_t rowMajorLayout = cublasLtOrder_t::CUBLASLT_ORDER_ROW;
    // Matrix A
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutCreate(&aLayout, std::get<1>(computeAndDataType),
                                   algoSelectorArgs[2], algoSelectorArgs[3],
                                   algoSelectorArgs[4]),
        "failed to create layout for matrix A");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &rowMajorLayout,
                                         sizeof(rowMajorLayout)),
        "failed to set layout order attribute for matrix A");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(aLayout,
                                         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &batchSize, sizeof(batchSize)),
        "failed to set batch size attribute for matrix A");
    int64_t batchStrideA = algoSelectorArgs[2] * algoSelectorArgs[3];
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(
            aLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batchStrideA,
            sizeof(batchStrideA)),
        "failed to set batch stride attribute for matrix A");

    MTRT_DBGF("%s", "[cuBLAS] created layout and set attributes for matrix A");

    // Matrix B
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutCreate(&bLayout, std::get<1>(computeAndDataType),
                                   algoSelectorArgs[7], algoSelectorArgs[8],
                                   algoSelectorArgs[9]),
        "failed to create layout for matrix B");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &rowMajorLayout,
                                         sizeof(rowMajorLayout)),
        "failed to set layout order attribute for matrix B");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(bLayout,
                                         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &batchSize, sizeof(batchSize)),
        "failed to set batch size attribute for matrix B");
    int64_t batchStrideB = algoSelectorArgs[7] * algoSelectorArgs[8];
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(
            bLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batchStrideB,
            sizeof(batchStrideB)),
        "failed to set batch stride attribute for matrix B");

    MTRT_DBGF("%s", "[cuBLAS] created layout and set attributes for matrix B");

    // Matrix C
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutCreate(&cLayout, std::get<1>(computeAndDataType),
                                   algoSelectorArgs[12], algoSelectorArgs[13],
                                   algoSelectorArgs[14]),
        "failed to create layout for matrix C");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &rowMajorLayout,
                                         sizeof(rowMajorLayout)),
        "failed to set layout order attribute for matrix C");
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(cLayout,
                                         CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &batchSize, sizeof(batchSize)),
        "failed to set batch size attribute for matrix C");
    int64_t batchStrideC = algoSelectorArgs[12] * algoSelectorArgs[13];
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatrixLayoutSetAttribute(
            cLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batchStrideC,
            sizeof(batchStrideC)),
        "failed to set batch stride attribute for matrix C");

    MTRT_DBGF("%s", "[cuBLAS] created layout and set attributes for matrix C");

    // Create matmul preference
    RETURN_ERROR_IF_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&preference),
                                 "failed to create matmul preference");
    MTRT_DBGF("%s", "[cuBLAS] created matmul preference");

    // Target CTA tile sizes if the M & N is valid (NOT 0)
    if (algoSelectorArgs[17] && algoSelectorArgs[18])
      targetTileSizes = std::make_unique<std::pair<int, int>>(
          std::make_pair<int, int>(algoSelectorArgs[17], algoSelectorArgs[18]));

    return getOkStatus();
  }

  // MxN tile size parameter
  // change the requestedAlgoCount to get all possible values
  // go through the result to get the closest
  Status selectAlgo(CublasLtHandle handle,
                    std::pair<int, int> targetTileSizes) {
    int returnedResults = 0;
    const int requestedAlgoCount = 32;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = {};
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulAlgoGetHeuristic(
            handle, operationDesc, aLayout, bLayout, cLayout, cLayout,
            preference, requestedAlgoCount, heuristicResult, &returnedResults),
        "failed to run algorithm selection");

    MTRT_DBGF("[cuBLAS] requested %d matmul algorithm heuristically",
              requestedAlgoCount);
    MTRT_DBGF("[cuBLAS] selected %d matmul algorithm heuristically",
              returnedResults);

    if (returnedResults == 0) {
      RETURN_ERROR_IF_CUBLAS_ERROR(CUBLAS_STATUS_NOT_SUPPORTED,
                                   "0 algorithms found by the algorithm "
                                   "selector. Unsupported problem");
    }

    MTRT_DBGF("%d algorithms found by the algorithm selector. Now filtering by "
              "tile size. ",
              returnedResults);

    int closestAlgoId = -1;
    double closestDistance = std::numeric_limits<int>::max();
    for (int algoIdx = 0; algoIdx < returnedResults; ++algoIdx) {
      size_t sizeWritten = 0;
      cublasLtMatmulAlgo_t matmalAlgo = heuristicResult[algoIdx].algo;
      RETURN_ERROR_IF_CUBLAS_ERROR(
          cublasLtMatmulAlgoCapGetAttribute(
              &matmalAlgo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten),
          "failed to query number of supported tile IDs for the selected "
          "algorithm.");

      // Number of tiles available to choose
      int nbTiles = int(sizeWritten / sizeof(int32_t));

      MTRT_DBGF("%d tile ids are supported for %d selected algorithm.", nbTiles,
                algoIdx);

      // No tile IDs if sizeWritten == 0
      if (!nbTiles)
        continue;

      std::vector<int32_t> tileA(nbTiles);

      RETURN_ERROR_IF_CUBLAS_ERROR(
          cublasLtMatmulAlgoCapGetAttribute(
              &matmalAlgo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA.data(),
              sizeof(int32_t) * nbTiles, &sizeWritten),
          "failed to query tile IDs for the selected algorithm.");

      for (int tileIdx = 0; tileIdx < nbTiles; ++tileIdx) {
        MTRT_DBGF("result tile: %d x %d\n", matmulTileIds[tileIdx].first,
                  matmulTileIds[tileIdx].second);
        double currentDistance = euclideanDistance(
            matmulTileIds[tileIdx].first, matmulTileIds[tileIdx].second,
            targetTileSizes.first, targetTileSizes.second);
        if (currentDistance < closestDistance)
          closestAlgoId = algoIdx;
      }
    }

    if (closestAlgoId == -1)
      RETURN_ERROR_IF_CUBLAS_ERROR(CUBLAS_STATUS_NOT_SUPPORTED,
                                   "0 algorithms found by the algorithm "
                                   "selector. Unsupported problem");

    MTRT_DBGF("The No. %d algorithm is picked as the best algorithm. ",
              closestAlgoId);
    algo = heuristicResult[closestAlgoId];

    return getOkStatus();
  }

  Status selectAlgo(CublasLtHandle handle) {
    // use restricted tile sizes if possible
    if (targetTileSizes)
      return selectAlgo(handle, *targetTileSizes);

    // Chose the first algo if there is no tile sizes restrictions
    int returnedResults = 0;
    int64_t requestedAlgoCount = 1;
    RETURN_ERROR_IF_CUBLAS_ERROR(
        cublasLtMatmulAlgoGetHeuristic(
            handle, operationDesc, aLayout, bLayout, cLayout, cLayout,
            preference, requestedAlgoCount, &algo, &returnedResults),
        "failed to run algorithm selection");
    if (returnedResults == 0) {
      RETURN_ERROR_IF_CUBLAS_ERROR(
          CUBLAS_STATUS_NOT_SUPPORTED,
          "0 algorithms found by the algorithm selector. Unsupported problem");
    }
    MTRT_DBGF("[cuBLAS] selected %d matmul algorithm heuristically",
              returnedResults);
    return getOkStatus();
  }

  cublasLtMatmulHeuristicResult_t algo;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatrixLayout_t aLayout = nullptr;
  cublasLtMatrixLayout_t bLayout = nullptr;
  cublasLtMatrixLayout_t cLayout = nullptr;
  cublasLtMatmulDesc_t operationDesc = nullptr;
  std::unique_ptr<std::pair<int, int>> targetTileSizes = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
// CuBLAS runtime functions
//===----------------------------------------------------------------------===//

// Run D = alpha*(A@B) + beta*C, where D=C (inplace update)
static void runCublasGemm(sol::this_state state, CublasLtHandle handle,
                          CudaStream stream, AlgoSelectResult &algo,
                          AllocTracker &tracker, sol::variadic_args varArgs) {
  MTRT_DBGF("%s", "[cuBLAS] executing gemm");
  auto scalarPointerMode = cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE;
  cublasStatus_t ptrModeStatus = cublasLtMatmulDescSetAttribute(
      algo.getMatmulDesc(), CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &scalarPointerMode, sizeof(scalarPointerMode));
  if (ptrModeStatus != CUBLAS_STATUS_SUCCESS)
    luaL_error(state,
               "[cuBLAS] failed to set cuBLAS matmul descriptor attribute: "
               "CUBLASLT_MATMUL_DESC_POINTER_MODE, with status: ",
               ptrModeStatus);
  auto workspace = allocate(tracker, PointerType::device,
                            algo.getAlgo().workspaceSize, std::nullopt, stream);
  MTRT_DBGF("[cuBLAS] Allocated %zd device bytes for workspace at: %lu",
            algo.getAlgo().workspaceSize, workspace->ptr);
  SET_LUA_ERROR_IF_ERROR(workspace, state);
  SET_LUA_ERROR_IF_CUBLAS_ERROR(
      cublasLtMatmul(handle, algo.getMatmulDesc(),
                     reinterpret_cast<void *>(varArgs[0].get<uintptr_t>()),
                     reinterpret_cast<void *>(varArgs[1].get<uintptr_t>()),
                     algo.getALayout(),
                     reinterpret_cast<void *>(varArgs[2].get<uintptr_t>()),
                     algo.getBLayout(),
                     reinterpret_cast<void *>(varArgs[3].get<uintptr_t>()),
                     reinterpret_cast<void *>(varArgs[4].get<uintptr_t>()),
                     algo.getCLayout(),
                     reinterpret_cast<void *>(varArgs[4].get<uintptr_t>()),
                     algo.getCLayout(), &(algo.getAlgo().algo),
                     reinterpret_cast<void *>(workspace->ptr),
                     algo.getAlgo().workspaceSize,
                     reinterpret_cast<cudaStream_t>(stream)),
      state, "[cuBLAS] run gemm failed");
  MTRT_DBGF("%s", "[cuBLAS] successfully executed gemm");
}

// Run C += A@B
static void runCublasMatmul(sol::this_state state, CublasLtHandle handle,
                            CudaStream stream, AlgoSelectResult &algo,
                            AllocTracker &tracker, sol::variadic_args varArgs) {
  MTRT_DBGF("%s", "[cuBLAS] executing matmul");
  float alpha = 1.0;
  auto alphaHostPtr =
      allocate(tracker, PointerType::host, sizeof(alpha), 64, {});
  *reinterpret_cast<float *>(alphaHostPtr->ptr) = alpha;
  float beta = 1.0;
  auto betaHostPtr = allocate(tracker, PointerType::host, sizeof(beta), 64, {});
  *reinterpret_cast<float *>(betaHostPtr->ptr) = beta;
  // By default scalars are on the host.
  // Scalar type is by default same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE.
  // However, we don't know what compute type is at this point, thus we
  // explicitly set it to float.
  auto scalarType = cudaDataType_t::CUDA_R_32F;
  cublasStatus_t scalarTypeStatus = cublasLtMatmulDescSetAttribute(
      algo.getMatmulDesc(), CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scalarType,
      sizeof(scalarType));
  if (scalarTypeStatus != CUBLAS_STATUS_SUCCESS)
    luaL_error(state,
               "[cuBLAS] failed to set cuBLAS matmul descriptor attribute: "
               "CUBLASLT_MATMUL_DESC_SCALE_TYPE, with status: ",
               scalarTypeStatus);
  auto workspace = allocate(tracker, PointerType::device,
                            algo.getAlgo().workspaceSize, std::nullopt, stream);
  MTRT_DBGF("[cuBLAS] Allocated %zd device bytes for workspace at: %lu",
            algo.getAlgo().workspaceSize, workspace->ptr);
  SET_LUA_ERROR_IF_ERROR(workspace, state);
  SET_LUA_ERROR_IF_CUBLAS_ERROR(
      cublasLtMatmul(
          handle, algo.getMatmulDesc(),
          reinterpret_cast<void *>(alphaHostPtr->ptr),
          reinterpret_cast<void *>(varArgs[0].get<uintptr_t>()),
          algo.getALayout(),
          reinterpret_cast<void *>(varArgs[1].get<uintptr_t>()),
          algo.getBLayout(), reinterpret_cast<void *>(betaHostPtr->ptr),
          reinterpret_cast<void *>(varArgs[2].get<uintptr_t>()),
          algo.getCLayout(),
          reinterpret_cast<void *>(varArgs[2].get<uintptr_t>()),
          algo.getCLayout(), &(algo.getAlgo().algo),
          reinterpret_cast<void *>(workspace->ptr),
          algo.getAlgo().workspaceSize, reinterpret_cast<cudaStream_t>(stream)),
      state, "[cuBLAS] run matmul failed");
  MTRT_DBGF("%s", "[cuBLAS] successfully executed matmul");
}

static void registerExecutorCuBLASModuleLuaRuntimeMethods(
    lua_State *state, AllocTracker *allocTracker,
    ResourceTracker *resourceTracker) {
  sol::state_view lua(state);
  lua["__cuda_blas_handle_create"] =
      [resourceTracker](sol::this_state state) -> CublasLtHandle {
    ADD_CUDA_MODULE_RANGE("cuda_blas_handle_create");
    StatusOr<CublasLtHandle> handle = CublasLtHandle::create(*resourceTracker);
    SET_LUA_ERROR_IF_ERROR(handle, state);
    return *handle;
  };

  lua["__cuda_blas_handle_destroy"] = [resourceTracker](sol::this_state state,
                                                        CublasLtHandle handle) {
    SET_LUA_ERROR_IF_CUBLAS_ERROR(cublasLtDestroy(handle), state,
                                  "failed to destroy cublasLt handle");
    resourceTracker->untrack(handle);
  };

  lua["__cuda_blas_algo_select"] =
      [](sol::this_state state, CublasLtHandle handle, int64_t dtype,
         int64_t batchSize, int64_t size0A, int64_t size1A, int64_t stride0A,
         int64_t stride1A, int64_t transposeA, int64_t size0B, int64_t size1B,
         int64_t stride0B, int64_t stride1B, int64_t transposeB, int64_t size0C,
         int64_t size1C, int64_t stride0C, int64_t stride1C, int64_t transposeC,
         int64_t tileSizesM,
         int64_t tileSizesN) -> std::unique_ptr<AlgoSelectResult> {
    ADD_CUDA_MODULE_RANGE("cuda_blas_algo_select");

    MTRT_DBGF("%s", "[cuBLAS] selecting matmul algorithm");
    MTRT_DBGF("Target CTA tile sizes: %d x %d", int(tileSizesM),
              int(tileSizesN));
    llvm::SmallVector<int64_t> args = {
        dtype,    batchSize,  size0A,     size1A,    stride0A,
        stride1A, transposeA, size0B,     size1B,    stride0B,
        stride1B, transposeB, size0C,     size1C,    stride0C,
        stride1C, transposeC, tileSizesM, tileSizesN};

    StatusOr<std::unique_ptr<AlgoSelectResult>> algo =
        AlgoSelectResult::get(handle, args);
    SET_LUA_ERROR_IF_ERROR(algo, state);
    return std::move(*algo);
  };

  lua["__cuda_blas_run_gemm"] =
      [allocTracker](sol::this_state state, CublasLtHandle handle,
                     CudaStream stream, AlgoSelectResult *algo,
                     sol::variadic_args varArgs) {
        ADD_CUDA_MODULE_RANGE("cuda_blas_run_gemm");
        assert((varArgs.size() == 3 || varArgs.size() == 5) &&
               "[cuBLAS] expected either 3 (MatMul) or 5 data pointers(GEMM).");
        if (varArgs.size() == 3)
          runCublasMatmul(state, handle, stream, *algo, *allocTracker, varArgs);
        else
          runCublasGemm(state, handle, stream, *algo, *allocTracker, varArgs);
      };
}

namespace mtrt {
void registerLuaCublasRuntimeExtension() {
  registerLuaRuntimeExtension(
      "cublas",
      LuaRuntimeExtension{[](const LuaRuntimeExtensionInitArgs &args) {
        registerExecutorCuBLASModuleLuaRuntimeMethods(
            args.state, args.allocTracker, args.resourceTracker);
      }});
}

} // namespace mtrt
