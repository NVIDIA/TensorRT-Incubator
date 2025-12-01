if(MLIR_TRT_ENABLE_WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Wno-error")
endif()

function(append_c_cxx_flags)
  string(JOIN " " flags ${ARGN})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags}" PARENT_SCOPE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flags}" PARENT_SCOPE)
endfunction()

# Enable additional warnings and runtime checks recommended
# by https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html
if(MLIR_TRT_ENABLE_EXTRA_CHECKS AND
   (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  # FORTIFY_SOURCE=3 requires O1 or higher.
  # GCC has default FORTIFY_SOURCE=2.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
     (NOT CMAKE_BUILD_TYPE STREQUAL "Debug"))
    add_compile_definitions(
      _FORTIFY_SOURCE=3
    )
  endif()

  # Note: '-D_GLIBCXX_ASSERTIONS' already added by
  # the 'HandleLLVMOptions'.

  append_c_cxx_flags(
    # Enforce that all global functions have declarations. This catches
    # missing `static` qualifiers on functions which are intended to be TU-local.
    -Wmissing-declarations
    -Wmissing-prototypes
    -Wunused
    # Make template instantiation errors more readable.
    -fdiagnostics-show-template-tree
    # Enable better bounds checking for trailing array members.
    # Value is 0 to 3, with 3 being the most strict. LLVM has some code that
    # only allows us to use 1 here.
    -fstrict-flex-arrays=1
    # Enable runtime checks for variable-size stack allocation validity.
    -fstack-clash-protection
    # Enable runtime checks for stack-based buffer overflows.
    -fstack-protector-strong
    # Treat obsolete C constructs as errors.
    -Werror=implicit
    -Werror=incompatible-pointer-types
    -Werror=int-conversion
  )

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR
     CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    append_c_cxx_flags(
      -fcf-protection=full
    )
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    append_c_cxx_flags(
      -Wself-assign
    )
  endif()
endif()

# The CUDA headers will trigger excessive "C++20 extensions warning" with
# clang19+ in macro usage.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wc++20-extensions"
  CXX_SUPPORTS_C20_EXTENSIONS_FLAG)
if(CXX_SUPPORTS_C20_EXTENSIONS_FLAG)
  add_compile_options(-Wno-c++20-extensions)
endif()
