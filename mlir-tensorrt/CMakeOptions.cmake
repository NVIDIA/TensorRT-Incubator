mtrt_option(MLIR_TRT_ENABLE_HLO "Whether to include stablehlo features" ON)
mtrt_option(MLIR_TRT_TARGET_TENSORRT "Enable exporting TensorRT dialect IR to a TensorRT engine" ON)
mtrt_option(MLIR_TRT_WITH_ASAN "Enable AddressSanitizer use" OFF)
mtrt_option(MLIR_TRT_ENABLE_PYTHON "Enable building the mlir_tensorrt_compiler python package." ON)
mtrt_option(MLIR_TRT_ENABLE_TESTING "Enable building optional tests" ON)
mtrt_option(MLIR_TRT_ENABLE_TORCH "Whether to include torch-mlir features" OFF)
mtrt_option(MLIR_TRT_ENABLE_NCCL "Enable the NCCL runtime module" OFF)
mtrt_option(MLIR_TRT_ENABLE_CUBLAS "Enable CUBLAS in the executor" ON)
mtrt_option(MLIR_TRT_ENABLE_SHARDY "Enable compiler build with Shardy" OFF)
mtrt_option(MLIR_TRT_ENABLE_CUDA "Enable the use of CUDA runtime" ON)
mtrt_option(MLIR_TRT_ENABLE_CUDATILE "Enable the use of CUDA Tile IR" ON)
mtrt_option(MLIR_TRT_ENABLE_MPI "Enable use of MPI in the runtime" ${MLIR_TRT_ENABLE_NCCL})
mtrt_option(MLIR_TRT_LINK_MTRT_DYLIB "Link all tools against libMTRT dylib" OFF)
mtrt_option(MLIR_TRT_LINK_MLIR_DYLIB "Use the libMLIR dylib to provide MLIR-TensorRT's MLIR dependencies" OFF)

mtrt_option(MLIR_TRT_ENABLE_PJRT_JAX "Enable PJRT and JAX integration, which requires building/install of JAX from source." ON)

# Those options control how TensorRT is found or downloaded.
set(MLIR_TRT_TENSORRT_DIR "" CACHE STRING "Path to TensorRT install directory")
set(MLIR_TRT_DOWNLOAD_TENSORRT_VERSION "10.12" CACHE STRING
   "Version of TensorRT to download and use. It overrides MLIR_TRT_TENSORRT_DIR.")

# These options are not part of the "feature flags" set that are automatically added as
# compile definitions. They are only used within CMake logic.
option(MLIR_TRT_ENABLE_WERROR "Enable compiler build with -Werror" OFF)
option(MLIR_TRT_ENABLE_EXTRA_CHECKS "Enable additional warnings and runtime checks" OFF)

# This option controls whether we use file paths (relative to the top-level
# source directory) in logging and debug info. This is useful for a) builds in
# production where we don't want full paths to be used and b) to make logging
# more readable. Changing the option will generally invalidate your compiler
# cache for the project, though, so try to avoid turning it on and off.
#
# - Only works if compiler supports '-ffile-prefix-map'.
# - Only applied if MTRT is the top-level project.
#
mtrt_option(MLIR_TRT_RELATIVE_DEBUG_PATHS
                      "Use relative file paths in logging and debug info" ON)

# The 'python' directory must come last because it actually instantiates
# the compiler and runtime python packages in '<build-dir>/python_packages'
# whereas the definition of the source groups belonging to those packages
# are distributed across multiple sub-directories.
set(MLIR_TRT_ENABLE_PROJECTS_DEFAULT
  executor tensorrt kernel
)

list(APPEND MLIR_TRT_ENABLE_PROJECTS_DEFAULT "compiler")

if(MLIR_TRT_ENABLE_PJRT_JAX)
  list(APPEND MLIR_TRT_ENABLE_PROJECTS_DEFAULT "integrations/PJRT")
endif()

if(MLIR_TRT_ENABLE_PYTHON)
  list(APPEND MLIR_TRT_ENABLE_PROJECTS_DEFAULT "integrations/python")
endif()

set(MLIR_TRT_ENABLE_PROJECTS "${MLIR_TRT_ENABLE_PROJECTS_DEFAULT}" CACHE STRING "Projects to enable")

# We need this option since CMAKE_CUDA_ARCHITECTURES=native has some issues.
set(MLIR_TRT_CUDA_ARCHITECTURES "detect" CACHE STRING
  "Equivalent to CMAKE_CUDA_ARCHITECTURES, but use \"detect\" to "
  "specify that the compute capability should be taken by querying the driver.")

# Set `MLIR_TRT_CUDA_TARGET` to `CUDA::cudart` if CUDA is enabled, otherwise set it to an empty string.
# TODO: consider adding an option to link cudart statically.
if(MLIR_TRT_ENABLE_CUDA)
  set(MLIR_TRT_CUDA_TARGET "CUDA::cudart" CACHE INTERNAL "")
else()
  set(MLIR_TRT_CUDA_TARGET "" CACHE INTERNAL "")
endif()

# Set `MLIR_TRT_MPI_TARGET` to `MPI::MPI_C` if MPI is enabled, otherwise set it to an empty string.
if(MLIR_TRT_ENABLE_MPI)
  set(MLIR_TRT_MPI_TARGET "MPI::MPI_C" CACHE INTERNAL "")
else()
  set(MLIR_TRT_MPI_TARGET "" CACHE INTERNAL "")
endif()

# Set `MLIR_TRT_NCCL_TARGET` to `NCCL` if NCCL is enabled, otherwise set it to an empty string.
if(MLIR_TRT_ENABLE_NCCL)
  set(MLIR_TRT_NCCL_TARGET "NCCL" CACHE INTERNAL "")
else()
  set(MLIR_TRT_NCCL_TARGET "" CACHE INTERNAL "")
endif()

