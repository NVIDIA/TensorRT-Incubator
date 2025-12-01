# This is non-cross compiling toolchain config for using
# the default clang/lld installed on the system.

# Use clang
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

# Use LLD
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")
