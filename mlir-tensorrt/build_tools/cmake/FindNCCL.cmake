find_path(NCCL_INCLUDE_DIR NAMES nccl.h REQUIRED)
# libnccl_static.a for static
find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
find_library(NCCL_STATIC_LIBRARY NAMES nccl_static REQUIRED)

message(STATUS "Found NCCL headers in ${NCCL_INCLUDE_DIR}")
message(STATUS "Found NCCL library at ${NCCL_LIBRARY}")
message(STATUS "Found NCCL static library at ${NCCL_STATIC_LIBRARY}")

if(NOT TARGET NCCL)
  add_library(NCCL SHARED IMPORTED GLOBAL)
  add_library(NCCL_STATIC STATIC IMPORTED GLOBAL)
  set_property(TARGET NCCL PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARY}")
  set_property(TARGET NCCL_STATIC PROPERTY IMPORTED_LOCATION "${NCCL_STATIC_LIBRARY}")
  target_include_directories(NCCL INTERFACE $<BUILD_INTERFACE:${NCCL_INCLUDE_DIR}>)
  target_include_directories(NCCL_STATIC INTERFACE $<BUILD_INTERFACE:${NCCL_INCLUDE_DIR}>)
  add_library(NCCL::NCCL ALIAS NCCL)
  add_library(NCCL::NCCL_STATIC ALIAS NCCL_STATIC)
endif()
