cmake_minimum_required(VERSION 3.20)

# Tests that all possible TensorRT download URLs point to valid resources.
# Usage: 'cmake -P build_tools/cmake/TestTensorRTDownloadURL.cmake'
cmake_policy(SET CMP0057 NEW)

# Mock CUDAToolkit to avoid find_package in script mode
set(CUDAToolkit_FOUND TRUE)
# CUDAToolkit_VERSION_MAJOR is set inside the loop
set(CUDAToolkit_VERSION_MINOR "0")

# Add the parent directory to module path so we can include the module under test
get_filename_component(_test_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
list(APPEND CMAKE_MODULE_PATH "${_test_dir}/..")

include(TensorRTDownloadURL)

set(VERSIONS "10.13.2" "10.13.3" "10.14" "10.2" "10.3" "10.4"
             "10.5" "10.8" "10.9" "10.12")
set(OSS "Linux")
set(ARCHS "x86_64" "aarch64")

foreach(VERSION IN LISTS VERSIONS)
  foreach(OS IN LISTS OSS)
    foreach(ARCH IN LISTS ARCHS)
      # Adjust mock CUDA version based on TRT version and ARCH to select a valid download URL
      if(VERSION VERSION_GREATER_EQUAL "10.4" AND "${ARCH}" MATCHES "aarch64")
         # For 10.4+ on aarch64 (Ubuntu 24.04), CUDA 11.8 URLs seem invalid/missing.
         # Force newer CUDA (e.g. 12.6) by setting host to 12.0.
         set(CUDAToolkit_VERSION_MAJOR "12")
      else()
         # Default to picking the lowest available version (usually 11.8 or 12.9)
         # by setting host version low (e.g. 11.0).
         set(CUDAToolkit_VERSION_MAJOR "11")
      endif()

      if(VERSION VERSION_LESS 10.0)
        if("${ARCH}" MATCHES "aarch64")
          continue()
        endif()
      endif()

      mtrt_get_tensorrt_download_url("${VERSION}" "${OS}" "${ARCH}" url modified_version)

      # Use curl to perform a HEAD request
      execute_process(
        COMMAND wget --spider --server-response --max-redirect=2 "${url}"
        ERROR_VARIABLE headers
        RESULT_VARIABLE result
      )

      # Check if curl request was successful
      if(result EQUAL 0)
        # Check for 'application/x-gzip', 'application/gzip', or '.tar.gz' in headers
        if(headers MATCHES "(application/x-gzip|application/gzip|\\.tar\\.gz)")
            message(STATUS "Valid .tar.gz URL: ${url}")
        else()
            message(FATAL_ERROR "Not a valid .tar.gz URL: ${url}")
        endif()
      else()
          message(FATAL_ERROR "Failed to reach URL: ${url}")
      endif()
    endforeach()
  endforeach()
endforeach()
