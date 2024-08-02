# Logic to setup `ENABLE_*` sanitizer options. This file should only be included
# if MLIR-TRT is the top-level project. Therefore, we don't bother prefixing
# the options with `MLIR_TRT_` like other cache variables that we setup in the
# main CMakeLists.txt.
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
  message(WARNING
    "Building MLIR-TensorRT with AddressSanitizer enabled."
  )

  # Force a cache variable to set the relevant LLVM option, otherwise
  # the upstream tests might not be configured correctly.
  set(LLVM_USE_SANITIZER "Address" CACHE STRING "" FORCE)
  set(MLIR_TRT_WITH_ASAN "ON" CACHE BOOL "" FORCE)
endif()

if(MSVC)
  if(ENABLE_ASAN)
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    add_compile_options(/fsanitize=address /fsanitize-address-use-after-return)
  endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
  option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
  option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
  if(NOT APPLE)
    option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
  endif()
  if((ENABLE_ASAN AND (ENABLE_TSAN OR ENABLE_MSAN)) OR
     (ENABLE_LSAN AND (ENABLE_TSAN OR ENABLE_MSAN)) OR
     (ENABLE_TSAN AND ENABLE_MSAN))
    message(FATAL_ERROR
      "Invalid sanitizer combination:\n"
      " ENABLE_ASAN: ${ENABLE_ASAN}\n"
      " ENABLE_LSAN: ${ENABLE_LSAN}\n"
      " ENABLE_TSAN: ${ENABLE_TSAN}\n"
      " ENABLE_MSAN: ${ENABLE_MSAN}"
      )
  endif()
  add_compile_options(
    -fno-omit-frame-pointer
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
    $<$<BOOL:${ENABLE_LSAN}>:-fsanitize=leak>
    $<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>
    $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
    $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
    )
  add_link_options(
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
    $<$<BOOL:${ENABLE_LSAN}>:-fsanitize=leak>
    $<$<BOOL:${ENABLE_MSAN}>:-fsanitize=memory>
    $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
    $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
    )
endif()
