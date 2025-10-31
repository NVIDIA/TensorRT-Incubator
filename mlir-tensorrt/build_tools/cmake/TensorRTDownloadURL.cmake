
macro(mtrt_require_os REQUIRED)
  set(error_msg "Downloading TensorRT ${ARG_VERSION} via CMake is not yet supported for target ${OS_NAME} ${TARGET_ARCH}")
  if(NOT OS_NAME MATCHES "${REQUIRED}")
    message(FATAL_ERROR "${error_msg}")
  endif()
endmacro()

macro(mtrt_require_arch REQUIRED)
  set(error_msg "Downloading TensorRT ${ARG_VERSION} via CMake is not yet supported for target ${OS_NAME} ${TARGET_ARCH}")
  if(NOT TARGET_ARCH MATCHES "${REQUIRED}")
    message(FATAL_ERROR "${error_msg}")
  endif()
endmacro()

# Get the CUDA version tag used when creating the TensorRT download URL.
# Why have to rely on a hand-coded map from TRT version to CUDA version(s).
# If multiple CUDA major versions are available (e.g. 12.9 and 13.0), we
# check what version is used on the host CTK. Otherwise, if host CTK is
# not found, use the lower version.
function(mtrt_get_tensorrt_cuda_version trt_version out_var)
  if(NOT CUDAToolkit_VERSION_MAJOR)
    find_package(CUDAToolkit)
  endif()

  set(ctk_version "")
  if(CUDAToolkit_FOUND)
    set(ctk_version "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  endif()

  set(trt_available_cuda_versions "")
  if(trt_version VERSION_GREATER 10.2 AND
     trt_version VERSION_LESS    10.4)
    set(trt_available_cuda_versions "11.8;12.5")
  elseif(trt_version VERSION_GREATER 10.4 AND
         trt_version VERSION_LESS    10.8)
    set(trt_available_cuda_versions "11.8;12.6")
  elseif(trt_version VERSION_GREATER 10.7 AND
         trt_version VERSION_LESS    10.10)
    set(trt_available_cuda_versions "11.8;12.8")
  elseif(trt_version VERSION_GREATER 10.10 AND
         trt_version VERSION_LESS    10.13)
    set(trt_available_cuda_versions "11.8;12.9")
  elseif(trt_version VERSION_GREATER_EQUAL "10.13")
    set(trt_available_cuda_versions "12.9;13.0")
  else()
    message(FATAL_ERROR "Could not determine available CUDA versions for TensorRT version ${trt_version}")
  endif()

  set(selected_cuda_version "")
  if(ctk_version)
    foreach(available_version IN_LISTS trt_available_cuda_versions)
      if(ctk_version VERSION_LESS_EQUAL available_version)
        set(selected_cuda_version "${available_version}")
      endif()
    endforeach()
  endif()

  if(NOT selected_cuda_version)
    list(GET trt_available_cuda_versions 0 selected_cuda_version)
  endif()

  message(STATUS "Selected CUDA version tag for TensorRT ${trt_version} is ${selected_cuda_version}")
  set("${out_var}" "${selected_cuda_version}" PARENT_SCOPE)
endfunction()

function(mtrt_get_tensorrt_download_url ARG_VERSION OS_NAME TARGET_ARCH ARG_OUT_VAR VERSION_OUT_VAR)
  if((NOT ARG_VERSION) OR (NOT ARG_OUT_VAR))
    message(FATAL_ERROR "Usage: get_tensorrt_download_url(version url_output_var)")
  endif()

  if(ARG_VERSION VERSION_EQUAL "8.6")
    set(ARG_VERSION "8.6.1.6")
  endif()

  if(ARG_VERSION VERSION_EQUAL "9.0")
    set(ARG_VERSION "9.0.1.4")
  endif()

  if(ARG_VERSION VERSION_EQUAL "9.1")
    set(ARG_VERSION "9.1.0.4")
  endif()

  if(ARG_VERSION VERSION_EQUAL "9.2")
    set(ARG_VERSION "9.2.0.5")
  endif()

  # Canonicalize "10.0" version by setting it to the latest public TRT 10.0 version.
  if(ARG_VERSION VERSION_EQUAL "10.0")
    set(ARG_VERSION "10.0.1.6")
  endif()

  # Canonicalize "10.1" version by setting it to the latest public TRT 10.1 version.
  if(ARG_VERSION VERSION_EQUAL "10.1")
    set(ARG_VERSION "10.1.0.27")
  endif()
  # Canonicalize "10.2" version by setting it to the latest public TRT 10.2 version.
  if(ARG_VERSION VERSION_EQUAL "10.2")
    set(ARG_VERSION "10.2.0.19")
  endif()
  # Canonicalize "10.3" version by setting it to the latest public TRT 10.3 version.
  if(ARG_VERSION VERSION_EQUAL "10.3")
    set(ARG_VERSION "10.3.0.26")
  endif()
  # Canonicalize "10.4" version by setting it to the latest public TRT 10.4 version.
  if(ARG_VERSION VERSION_EQUAL "10.4")
    set(ARG_VERSION "10.4.0.26")
  endif()
  # Canonicalize "10.5" version by setting it to the latest public TRT 10.5 version.
  if(ARG_VERSION VERSION_EQUAL "10.5")
    set(ARG_VERSION "10.5.0.18")
  endif()
  # Canonicalize "10.6" version by setting it to the latest public TRT 10.6 version.
  if(ARG_VERSION VERSION_EQUAL "10.6")
    set(ARG_VERSION "10.6.0.26")
  endif()
  # Canonicalize "10.7" version by setting it to the latest public TRT 10.7 version.
  if(ARG_VERSION VERSION_EQUAL "10.7")
    set(ARG_VERSION "10.7.0.23")
  endif()
  if(ARG_VERSION VERSION_EQUAL "10.13")
    set(ARG_VERSION "10.13.0.35")
  endif()

  if(ARG_VERSION VERSION_EQUAL "10.8")
    set(ARG_VERSION "10.8.0.43")
  endif()

  if(ARG_VERSION VERSION_EQUAL "10.9")
    set(ARG_VERSION "10.9.0.34")
  endif()

  if(ARG_VERSION VERSION_EQUAL "10.12")
    set(ARG_VERSION "10.12.0.36")
  endif()

  if(ARG_VERSION VERSION_EQUAL "10.13.0.35")
    set(ARG_VERSION "10.13.0.35")
  endif()

  if(ARG_VERSION VERSION_EQUAL "10.14")
    set(ARG_VERSION "10.14.1.48")
  endif()

  set(downloadable_versions
    "8.6.1.6"
    "9.0.1.4" "9.1.0.4" "9.2.0.5"
    "10.0.0.6"
    "10.0.1.6"
    "10.1.0.27"
    "8.6.1.6"
    "10.2.0.19"
    "10.3.0.6"
    "10.3.0.26"
    "10.4.0.26"
    "10.5.0.18"
    "10.6.0.26"
    "10.7.0.23"
    "10.8.0.43"
    "10.9.0.34"
    "10.12.0.36"
    "10.13.0.35"
    "10.14.1.48"
  )

  if(NOT ARG_VERSION IN_LIST downloadable_versions)
    message(FATAL_ERROR "CMake download of TensorRT is only available for \
      the following versions: ${downloadable_versions}")
  endif()

  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" trt_short_version ${ARG_VERSION})

  # Get the CUDA version tag.
  mtrt_get_tensorrt_cuda_version("${ARG_VERSION}" TRT_CUDA_VERSION)

  # For aarch64, the published packages are only for
  # "Ubuntu-20.04". I believe this corresponds to NVIDIA supported ARM server
  # config (not Jetson). These builds may still work on embedded targets, but
  # the TRT libraries will not contain the right pre-compiled kernels, making
  # compilation times very long since just loading TRT may require JIT-compiling
  # many PTX kernels.
  if(TARGET_ARCH MATCHES "aarch64" AND TRT_CUDA_VERSION VERSION_LESS 13.0)
    message(WARNING "Automatic TensorRT package downloads for CUDA versions less than 13.0 for aarch64 are for"
    " ARM SBSA systems, not embedded systems. If you are using NVIDIA Jetson, NVIDIA Drive, "
    " or another embedded system you should download TensorRT manually from the appropriate source:\n"
    "For Drive, see: https://docs.nvidia.com/drive/\n"
    "For Jetson, see: https://developer.nvidia.com/embedded/jetpack")

    if(ARG_VERSION VERSION_LESS 10.0)
      set(OS_NAME "Ubuntu-20.04")
    elseif(ARG_VERSION VERSION_LESS 10.4)
      set(OS_NAME "Ubuntu-22.04")
    elseif(ARG_VERSION VERSION_LESS 10.7)
      set(OS_NAME "Ubuntu-24.04")
    else()
      set(OS_NAME "Linux")
    endif()
  endif()

  # We need OS_NAME and OS_NAME_LOWER because apparently it's impossible for NVIDIA
  # to publish URLs using a stable convention, and URLs for different versions have
  # different capitalization requirements.
  # set(OS_NAME "${CMAKE_SYSTEM_NAME}")
  string(TOLOWER "${OS_NAME}" OS_NAME_LOWER)

  if(TARGET_ARCH MATCHES "x86_64")
    mtrt_require_os("Linux")
  endif()

  if(TARGET_ARCH MATCHES "aarch64")
    if(ARG_VERSION VERSION_LESS 10.7)
      mtrt_require_os("Ubuntu")
    else()
      mtrt_require_os("Linux")
    endif()
  endif()

  # Handle TRT 8 versions.
  if(ARG_VERSION VERSION_LESS 9.0.0 AND ARG_VERSION VERSION_GREATER 8.0.0)
    set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/${trt_short_version}/tars/TensorRT-${ARG_VERSION}.${OS_NAME_LOWER}.${TARGET_ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz")
  endif()

  # Handle TensorRT 9 versions.
  if(ARG_VERSION VERSION_LESS 10.0.0 AND ARG_VERSION VERSION_GREATER 9.0.0)
    if(ARG_VERSION VERSION_LESS 9.2.0)
      mtrt_require_arch("x86_64")
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/${trt_short_version}/tars/tensorrt-${ARG_VERSION}.${OS_NAME_LOWER}.${TARGET_ARCH}-gnu.cuda-12.2.tar.gz")
    else()
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short_version}/tensorrt-${ARG_VERSION}.${OS_NAME_LOWER}.${TARGET_ARCH}-gnu.cuda-12.2.tar.gz")
    endif()
  else()
    # TensorRT 10 EA
    if(ARG_VERSION VERSION_EQUAL 10.0.0.6)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/tensorrt-10.0.0.6.${OS_NAME_LOWER}.${TARGET_ARCH}-gnu.cuda-12.4.tar.gz")
    # TensorRT 10.0 GA
    elseif(ARG_VERSION VERSION_EQUAL 10.0.1.6)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short_version}/tars/TensorRT-${ARG_VERSION}.${OS_NAME}.${TARGET_ARCH}-gnu.cuda-12.4.tar.gz")
    # TensorRT 10.1 GA
    elseif(ARG_VERSION VERSION_EQUAL 10.1.0.27)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short_version}/tars/tensorrt-${ARG_VERSION}.${OS_NAME_LOWER}.${TARGET_ARCH}-gnu.cuda-12.4.tar.gz")
    # TensorRT 10.2 GA
    elseif(ARG_VERSION VERSION_EQUAL 10.2.0.19)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.2.0/tars/TensorRT-10.2.0.19.${OS_NAME}.${TARGET_ARCH}-gnu.cuda-12.5.tar.gz")
    # TensorRT 10.3 GA
    elseif(ARG_VERSION VERSION_EQUAL 10.3.0.26)
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-${ARG_VERSION}.${OS_NAME}.${TARGET_ARCH}-gnu.cuda-12.5.tar.gz")
    # Other TensorRT versions.
    else()
      set(_url "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_short_version}/tars/TensorRT-${ARG_VERSION}.${OS_NAME}.${TARGET_ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz")
    endif()
  endif()

  if(NOT _url)
    message(FATAL_ERROR "Downloading TensorRT ${ARG_VERSION} via CMake is not yet supported for target ${OS_NAME} ${TARGET_ARCH}")
  endif()

  set("${ARG_OUT_VAR}" "${_url}" PARENT_SCOPE)
  set("${VERSION_OUT_VAR}" "${ARG_VERSION}" PARENT_SCOPE)
endfunction()
