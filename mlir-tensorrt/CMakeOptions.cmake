mtrt_option(MLIR_TRT_ENABLE_HLO "Whether to include stablehlo features" ON)
mtrt_option(MLIR_TRT_ENABLE_ASSERTIONS "Enables assertions" ON)
mtrt_option(MLIR_TRT_TARGET_TENSORRT "Enable exporting TensorRT dialect IR to a TensorRT engine" ON)
mtrt_option(MLIR_TRT_WITH_ASAN "Enable AddressSanitizer use" OFF)
mtrt_option(MLIR_TRT_ENABLE_PYTHON "Enable building the mlir_tensorrt_compiler python package." ON)
mtrt_option(MLIR_TRT_ENABLE_TESTING "Enable building optional tests" ON)
mtrt_option(MLIR_TRT_ENABLE_TORCH "Whether to include torch-mlir features" OFF)
mtrt_option(MLIR_TRT_ENABLE_NCCL "Enable the NCCL runtime module" OFF)
mtrt_option(MLIR_TRT_ENABLE_CUBLAS "Enable CUBLAS in the executor" ON)
mtrt_option(MLIR_TRT_ENABLE_CUDA "Enable the use of CUDA runtime" ON)
mtrt_option(MLIR_TRT_ENABLE_MPI "Enable use of MPI in the runtime" ${MLIR_TRT_ENABLE_NCCL})
mtrt_option(MLIR_TRT_ENABLE_SHARDY "Enable use of Shardy" OFF)

mtrt_option(MLIR_TRT_LINK_MTRT_DYLIB "Link all tools against libMTRT dylib" OFF)
mtrt_option(MLIR_TRT_LINK_MLIR_DYLIB "Use the libMLIR dylib to provide MLIR-TensorRT's MLIR dependencies" OFF)

# The 'python' directory must come last because it actually instantiates
# the compiler and runtime python packages in '<build-dir>/python_packages'
# whereas the definition of the source groups belonging to those packages
# are distributed across multiple sub-directories.
set(MLIR_TRT_ENABLE_PROJECTS_DEFAULT
  executor tensorrt compiler integrations/python
)

if(NOT MLIR_TRT_ENABLE_PYTHON)
  list(REMOVE_ITEM MLIR_TRT_ENABLE_PROJECTS_DEFAULT "integrations/python")
endif()

set(MLIR_TRT_ENABLE_PROJECTS ${MLIR_TRT_ENABLE_PROJECTS_DEFAULT} CACHE STRING "Projects to enable")

set(MLIR_TRT_TENSORRT_DIR "" CACHE STRING "Path to TensorRT install directory")
set(MLIR_TRT_DOWNLOAD_TENSORRT_VERSION "10.13" CACHE STRING
   "Version of TensorRT to download and use. It overrides MLIR_TRT_TENSORRT_DIR.")
set(MLIR_TRT_PACKAGE_CACHE_DIR "" CACHE STRING "Directory where to cache downloaded C++ packages")
# We need this option since CMAKE_CUDA_ARCHITECTURES=native has some issues.
set(MLIR_TRT_CUDA_ARCHITECTURES "detect" CACHE STRING
  "Equivalent to CMAKE_CUDA_ARCHITECTURES, but use \"detect\" to "
  "specify that the compute capability should be taken by querying the driver.")

if(MLIR_TRT_ENABLE_CUDA)
  set(MLIR_TRT_CUDA_TARGET "CUDA::cudart" CACHE INTERNAL "")
else()
  set(MLIR_TRT_CUDA_TARGET "" CACHE INTERNAL "")
endif()

if(MLIR_TRT_ENABLE_MPI)
  set(MLIR_TRT_MPI_TARGET "MPI::MPI_C" CACHE INTERNAL "")
else()
  set(MLIR_TRT_MPI_TARGET "" CACHE INTERNAL "")
endif()

if(MLIR_TRT_ENABLE_NCCL)
  set(MLIR_TRT_NCCL_TARGET "NCCL" CACHE INTERNAL "")
else()
  set(MLIR_TRT_NCCL_TARGET "" CACHE INTERNAL "")
endif()
