# Utilities for declaring and downloading dependencies.
#-------------------------------------------------------------------------------------
# Calculate a gitlab source archive download URL from a project ID and commit ID.
# The URL is stored in `urlVar` and the auth HTTP header in `headerVar`.
#-------------------------------------------------------------------------------------
function(nv_register_package)
  cmake_parse_arguments(ARG
    "" "NAME" "" ${ARGN})
  set(NV_PACKAGE_${ARG_NAME}_ARGS ${ARGV} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------------
# Calculate a gitlab source archive download URL from a project ID and commit ID.
# The URL is stored in `urlVar` and the auth HTTP header in `headerVar`.
#-------------------------------------------------------------------------------------
macro(nv_add_package name)
  cmake_parse_arguments(_vap "" "POST_ADD_HOOK;PRE_ADD_HOOK;DOWNLOAD_NAME;NAME" "" ${NV_PACKAGE_${name}_ARGS})
  if(_vap_PRE_ADD_HOOK)
    cmake_language(EVAL CODE "${_vap_PRE_ADD_HOOK}")
  endif()
  if(_vap_DOWNLOAD_NAME)
    mlir_tensorrt_aCPdd_package(
      NAME ${_vap_DOWNLOAD_NAME}
      ${_vap_UNPARSED_ARGUMENTS})
    set(${NAME}_SOURCE_DIR
      ${_vap_DOWNLOAD_NAME}_SOURCE_DIR)
    set(${NAME}_BINARY_DIR
      ${_vap_DOWNLOAD_NAME}_BINARY_DIR)
  else()
    mlir_tensorrt_add_package(
      NAME ${_vap_NAME}
      ${_vap_UNPARSED_ARGUMENTS})
  endif()

  if(_vap_POST_ADD_HOOK)
    cmake_language(EVAL CODE "${_vap_POST_ADD_HOOK}")
  endif()
endmacro()

#-------------------------------------------------------------------------------------
# Wrapper around CPMAddPackage This functions exactly like CPMAddPackage.
#-------------------------------------------------------------------------------------
macro(mlir_tensorrt_add_package)
  cmake_parse_arguments(ARG
    "" "GIT_REPOSITORY;GIT_TAG" "" ${ARGN})

  if(ARG_GIT_TAG)
    set(_mtrt_tmp_CACHE_KEY "${ARG_GIT_TAG}")
  endif()

  if(ARG_GIT_REPOSITORY)
    list(APPEND ARG_UNPARSED_ARGUMENTS
      GIT_REPOSITORY "${ARG_GIT_REPOSITORY}"
      )
  endif()
  if(ARG_GIT_TAG)
    list(APPEND ARG_UNPARSED_ARGUMENTS
      GIT_TAG "${ARG_GIT_TAG}"
      )
  endif()

  if(_mtrt_tmp_CACHE_KEY)
    list(APPEND ARG_UNPARSED_ARGUMENTS
         CUSTOM_CACHE_KEY "${_mtrt_tmp_CACHE_KEY}")
    set(_mtrt_tmp_CACHE_KEY)
  endif()

  CPMAddPackage(
    ${ARG_UNPARSED_ARGUMENTS}
  )
endmacro()
