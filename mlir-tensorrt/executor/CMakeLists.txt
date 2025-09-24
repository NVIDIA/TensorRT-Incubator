include(cmake/ExecutorFunctions.cmake)

find_package(NVTX REQUIRED)
find_package(Lua REQUIRED)
find_package(Sol2 REQUIRED)
find_package(Flatbuffers REQUIRED)

if(MLIR_TRT_ENABLE_CUDA)
  mlir_executor_find_and_patch_libnvptxcompiler(CUDANVPTXCompilerLibraryPatched)
endif()

# Check that the MLIRTensorRTCommon project provided the required targets
# when TensorRT is enabled.
if(MLIR_TRT_TARGET_TENSORRT AND NOT TARGET MLIRTRTTensorRTDynamicLoader)
  message(FATAL_ERROR "MLIR_TRT_TARGET_TENSORRT=ON but the MLIRTRTTensorRTDynamicLoader library was not found")
endif()

if(MLIR_TRT_ENABLE_MPI)
  find_package(MPI COMPONENTS C)
endif()

if(MLIR_TRT_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
endif()

if(MLIR_TRT_ENABLE_NCCL AND NOT TARGET NCCL)
  message(FATAL_ERROR "MLIR_TRT_ENABLE_NCCL=ON but NCCL was not found")
endif()
if(MLIR_TRT_ENABLE_MPI AND NOT MPI_FOUND)
  message(FATAL_ERROR "MLIR_TRT_ENABLE_MPI=ON but MPI was not found")
endif()

#-------------------------------------------------------------------------------
# Project Targets
#-------------------------------------------------------------------------------
add_library(MLIRExecutorIncludes INTERFACE)
target_include_directories(MLIRExecutorIncludes INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

set(MLIR_EXECUTOR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_EXECUTOR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MLIR_EXECUTOR_SOURCE_DIR}/include)
include_directories(${MLIR_EXECUTOR_BINARY_DIR}/include)

add_subdirectory(include/mlir-executor)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

mtrt_get_project_targets(MLIRExecutor OUT_VAR MLIR_EXECUTOR_LIBS)
add_library(MLIRTensorRTExecutorLibraries INTERFACE)
target_link_libraries(MLIRTensorRTExecutorLibraries INTERFACE
  ${MLIR_EXECUTOR_LIBS})
add_library(MLIRTensorRT::Executor ALIAS MLIRTensorRTExecutorLibraries)
add_mlir_library_install(MLIRTensorRTExecutorLibraries)
