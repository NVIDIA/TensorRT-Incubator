cmake_minimum_required(VERSION 3.20)

# Add the parent directory to module path so we can include the module under test
get_filename_component(_test_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
list(APPEND CMAKE_MODULE_PATH "${_test_dir}/..")

include(TensorRTDownloadURL)

# Helper function to run a test case
function(run_test trt_version ctk_major ctk_minor target_arch expected_result)
  # Setup mock environment
  if("${ctk_major}" STREQUAL "")
    # Simulate CUDAToolkit not found
    set(CUDAToolkit_FOUND FALSE)
    # # We set MAJOR to a dummy value to prevent the function from calling find_package(CUDAToolkit)
    # # which would try to find the actual system CUDA.
    # # Note: CMake treats "0" as False, so we must use a non-zero value to skip the `if(NOT ...)` check.
    # set(CUDAToolkit_VERSION_MAJOR "999")
    # set(CUDAToolkit_VERSION_MINOR "0")
  else()
    # Simulate CUDAToolkit found with specific version
    set(CUDAToolkit_FOUND TRUE)
    set(CUDAToolkit_VERSION_MAJOR "${ctk_major}")
    set(CUDAToolkit_VERSION_MINOR "${ctk_minor}")
  endif()

  # Call the function under test
  mtrt_get_tensorrt_cuda_version("${trt_version}" "${target_arch}" result)

  # Check result
  if(NOT "${result}" STREQUAL "${expected_result}")
    if("${ctk_major}" STREQUAL "")
      set(ctk_str "Not Found")
    else()
      set(ctk_str "${ctk_major}.${ctk_minor}")
    endif()
    message(FATAL_ERROR "Test Failed for TRT ${trt_version} with Host CUDA ${ctk_str}.\nExpected: ${expected_result}\nActual: ${result}")
  else()
    message(STATUS "Test Passed: TRT ${trt_version} + CUDA ${ctk_major}.${ctk_minor} -> ${result}")
  endif()
endfunction()

message(STATUS "Running TestTensorRTCUDAVersion...")

# Test Group 1: TRT 10.3 (Available: 11.8, 12.5)
# Logic:
#   If host <= available, set selected = available.
#   List is iterated in order.
#   If multiple match, the last one matching is kept (largest available that is >= host).

# Case 1.1: No CUDA found. Should pick last (12.5).
run_test("10.3" "" "" "x86_64" "12.5")

# Case 1.2: Host CUDA 11.8.
# 11.8 <= 11.8 (True -> 11.8)
# Break.
run_test("10.3" "11" "8" "x86_64" "11.8")

# Case 1.3: Host CUDA 12.0.
# 12.0 <= 11.8 (False)
# 12.0 <= 12.5 (True -> 12.5)
run_test("10.3" "12" "0" "x86_64" "12.5")

# Case 1.4: Host CUDA 12.6.
# 12.6 <= 11.8 (False)
# 12.6 <= 12.5 (False)
# Fallback to last (12.5) because host > max available
run_test("10.3" "12" "6" "x86_64" "12.5")

# Test Group 2: TRT 10.5 (Available: 11.8, 12.6)
# Host 12.6 -> 12.6 <= 12.6 (True -> 12.6)
run_test("10.5" "12" "6" "x86_64" "12.6")

# Test Group 3: TRT 10.8 (Available: 11.8, 12.8)
# Host 12.7 -> 12.7 <= 12.8 (True -> 12.8)
run_test("10.8" "12" "7" "x86_64" "12.8")

# Test Group 4: TRT 10.11 (Available: 11.8, 12.9)
# Host 12.9 -> 12.9 <= 12.9 (True -> 12.9)
run_test("10.11" "12" "9" "x86_64" "12.9")

# Test Group 5: TRT 10.13 (Available: 11.8, 12.9) because < 10.13.2
# Host 12.0 -> 12.0 <= 12.9 (True -> 12.9)
run_test("10.13" "12" "0" "x86_64" "12.9")

# Host 13.1 -> Fallback 12.9 (last available)
run_test("10.13" "13" "1" "x86_64" "12.9")

message(STATUS "All tests passed successfully.")

