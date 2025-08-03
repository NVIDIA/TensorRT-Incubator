# ------------------------------------------------------------------------------
# Enable use of ccache for C, CXX, and CUDA.
# - This is only done if MLIR-TensorRT is the top-level project.
# - Not finding ccache is not considered an error, but a warning is emitted.
# ------------------------------------------------------------------------------
function(mtrt_use_compiler_cache)
  # Do nothing if we are not the top-level project.
  if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    return()
  endif()

  find_program(CCACHE_EXECUTABLE ccache)
  if(NOT CCACHE_EXECUTABLE)
    message(WARNING
      "The 'ccache' program was not found; install ccache and re-configure "
      " to improve incremental build performance")
    return()
  endif()

  if(MSVC)
    message(WARNING
      "Support for automatic use of ccache "
      " with MSVC is not implemented."
      )
    return()
  endif()

  # Use a cache variable so the user can override this
  set(CCACHE_ENV
    "CCACHE_CPP2=true"
    CACHE STRING "List of environment variables for ccache, each in key=value form"
    )

  if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
    foreach(lang IN ITEMS C CXX CUDA)
      set(cmd ${CMAKE_COMMAND} -E env ${CCACHE_ENV} ${CCACHE_EXECUTABLE})
      set(CMAKE_${lang}_COMPILER_LAUNCHER
        ${cmd}
        PARENT_SCOPE
        )
      set(msg
        "Set CMAKE_${lang}_COMPILER_LAUNCHER=${cmd}")
      string(REPLACE ";" " " msg "${msg}")
      message(STATUS "${msg}")
    endforeach()
  else()
    message(WARNING
      "Automatic ccache configuration for "
      "generator ${CMAKE_GENERATOR} not implemented."
      )
  endif()
endfunction()
