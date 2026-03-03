# Toolchain for use with conda environment on Linux x86_64.
set(_conda_triple "x86_64-conda-linux-gnu")
set(conda_build_sysroot "$ENV{CONDA_PREFIX}/${_conda_triple}/sysroot")

# Common paths
set(CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX};${conda_build_sysroot}")
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}:${conda_build_sysroot}")
set(CMAKE_PROGRAM_PATH "$ENV{CONDA_PREFIX}/bin:${conda_build_sysroot}/bin")

set(CMAKE_AR "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ar")
set(CMAKE_C_COMPILER "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-clang")
set(CMAKE_C_COMPILER_AR "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ar")
set(CMAKE_C_COMPILER_RANLIB "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ranlib")
set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-clang++")
set(CMAKE_CXX_COMPILER_AR "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ar")
set(CMAKE_CXX_COMPILER_RANLIB "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ranlib")
set(CMAKE_RANLIB "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-ranlib")
set(CMAKE_STRIP "$ENV{CONDA_PREFIX}/bin/${_conda_triple}-strip")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=mold")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=mold")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=mold")
set(CMAKE_LINKER "$ENV{CONDA_PREFIX}/bin/mold")
