#-------------------------------------------------------------------------------------
# CMake policy settings for MLIR-TensorRT project
#
# This file centralizes all CMake policy decisions to ensure consistency across
# the project. Each policy setting includes documentation explaining why the
# specific choice was made.
#
# Policy settings are synchronized with LLVM project where applicable
# (see third_party/llvm-project/cmake/Modules/CMakePolicy.cmake)
#-------------------------------------------------------------------------------------

# CMP0114: ExternalProject step targets fully adopt their steps
# New in CMake 3.19: https://cmake.org/cmake/help/latest/policy/CMP0114.html
# Set to OLD to maintain compatibility with LLVM's ExternalProject usage patterns
if(POLICY CMP0114)
  cmake_policy(SET CMP0114 OLD)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20: https://cmake.org/cmake/help/latest/policy/CMP0116.html
# Set to OLD to maintain compatibility with LLVM's custom command patterns
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# CMP0141: MSVC debug information format flags are selected via
# CMAKE_MSVC_DEBUG_INFORMATION_FORMAT instead of embedding in CMAKE_CXX_FLAGS_*
# New in CMake 3.25: https://cmake.org/cmake/help/latest/policy/CMP0141.html
# Set to NEW to support debug info with SCCache (https://github.com/mozilla/sccache)
# and avoid "fatal error C1041: cannot open program database" errors
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

# CMP0144: find_package() uses uppercase <PackageName>_ROOT variables
# New in CMake 3.27: https://cmake.org/cmake/help/latest/policy/CMP0144.html
# Set to NEW to use modern variable naming conventions
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# CMP0156: De-duplicate libraries on link lines based on linker capabilities
# New in CMake 3.29: https://cmake.org/cmake/help/latest/policy/CMP0156.html
# Set to NEW to avoid "ld: warning: ignoring duplicate libraries" warnings,
# particularly important when building with the Apple linker
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()

# CMP0179: De-duplication of static libraries on link lines keeps first occurrence
# New in CMake 3.31: https://cmake.org/cmake/help/latest/policy/CMP0179.html
# Set to NEW to unify behavior across platforms and work around LLD ELF backend bug
# Note: This policy depends on CMP0156=NEW
if(POLICY CMP0179)
  cmake_policy(SET CMP0179 NEW)
endif()

# CMP0169: FetchContent_Populate direct usage
# New in CMake 3.30: https://cmake.org/cmake/help/latest/policy/CMP0169.html
# Set to OLD to allow direct FetchContent_Populate calls (used by CPM.cmake)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# CMP0140: return() checks for valid variable names
# New in CMake 3.25: https://cmake.org/cmake/help/latest/policy/CMP0140.html
# Set to NEW to enable stricter variable name validation in return() statements
if(POLICY CMP0140)
  cmake_policy(SET CMP0140 NEW)
endif()

# CMP0177: Normalize all DESTINATION values given in any form of the install()
# command, except for the INCLUDES DESTINATION of the install(TARGETS) form. The
# normalization performed is the same as for the cmake_path() command (see
# Normalization).
# New in CMake 3.31: https://cmake.org/cmake/help/latest/policy/CMP0177.html
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()
