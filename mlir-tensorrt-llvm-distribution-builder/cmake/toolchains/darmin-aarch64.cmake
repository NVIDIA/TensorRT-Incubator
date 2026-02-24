# Toolchain for use with conda environment on macOS arm64 (Apple Silicon).
set(_conda_triple "arm64-apple-darwin20.0.0")
set(conda_build_sysroot "$ENV{CONDA_PREFIX}/${_conda_triple}/sysroot")

# Common paths
set(CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX};${conda_build_sysroot}")
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}:${conda_build_sysroot}")
set(CMAKE_PROGRAM_PATH "$ENV{CONDA_PREFIX}/bin:${conda_build_sysroot}/bin")

# macOS: Use clang/clang++ directly (Apple toolchain)
set(CMAKE_C_COMPILER "$ENV{CONDA_PREFIX}/bin/clang")
set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/clang++")
set(CMAKE_AR "$ENV{CONDA_PREFIX}/bin/ar")
set(CMAKE_RANLIB "$ENV{CONDA_PREFIX}/bin/ranlib")
set(CMAKE_STRIP "$ENV{CONDA_PREFIX}/bin/strip")
