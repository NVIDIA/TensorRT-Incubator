# Usage:
#   mtrt_collect_interface_link_closure(OUT_LINKS OUT_INCS tgtA [tgtB ...]
#     [STOP_AT <func_name>]
#     [INCLUDE_FRONTIER])
#
# Returns (deduped, first-seen order):
#   OUT_LINKS:    transitive INTERFACE_LINK_LIBRARIES entries
#   OUT_INCS:     transitive INTERFACE_INCLUDE_DIRECTORIES along the same traversal
#
# Options:
#   STOP_AT <func_name>  - A CMake function taking a single argument (target
#     name) that sets '_is_frontier' in PARENT_SCOPE to TRUE/FALSE. When a
#     dependency is on the frontier, its own transitive dependencies are NOT
#     traversed.
#   INCLUDE_FRONTIER - When set, frontier nodes themselves are still included
#     in the returned sets (but their deps are not traversed). Without this
#     flag frontier nodes are excluded entirely.
#
function(mtrt_collect_interface_link_closure OUT_LINKS OUT_INCS)
  cmake_parse_arguments(_arg "INCLUDE_FRONTIER" "STOP_AT" "" ${ARGN})
  set(_roots "${_arg_UNPARSED_ARGUMENTS}")
  if (NOT _roots)
    message(FATAL_ERROR
      "mtrt_collect_interface_link_closure(OUT_LINKS OUT_INCS target [...]) needs at least one target")
  endif()

  set(_stop_at_func "${_arg_STOP_AT}")
  set(_include_frontier "${_arg_INCLUDE_FRONTIER}")

  function(_resolve_alias _in _out)
    if (TARGET "${_in}")
      get_target_property(_aliased "${_in}" ALIASED_TARGET)
      if (_aliased)
        set(${_out} "${_aliased}" PARENT_SCOPE)
      else()
        set(${_out} "${_in}" PARENT_SCOPE)
      endif()
    else()
      set(${_out} "${_in}" PARENT_SCOPE)
    endif()
  endfunction()

  function(_mangle_prop_key _tgt _out)
    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _m "${_tgt}")
    set(${_out} "${_m}" PARENT_SCOPE)
  endfunction()

  function(_get_tprop_list _tgt _prop _out)
    get_target_property(_v "${_tgt}" "${_prop}")
    if (_v STREQUAL "_v-NOTFOUND")
      unset(_v)
    endif()
    set(${_out} "${_v}" PARENT_SCOPE)
  endfunction()

  function(_check_frontier _tgt _func _out)
    set(_is_frontier FALSE)
    cmake_language(CALL "${_func}" "${_tgt}")
    set(${_out} "${_is_frontier}" PARENT_SCOPE)
  endfunction()

  # Post-order collector with memoization.
  # _stop_func: name of the frontier-check function, or "" for none
  # _incl_frontier: TRUE if frontier nodes should be included
  #
  # Memoization is scoped to a single top-level call — the caller clears the
  # cache after the traversal completes so that successive calls with different
  # options never see stale results.
  function(_collect_target _tgt _stop_func _incl_frontier OUT_L OUT_I)
    _mangle_prop_key("${_tgt}" _key)

    get_property(_cachedL GLOBAL PROPERTY "_iface_link_closure_${_key}" SET)
    get_property(_cachedI GLOBAL PROPERTY "_iface_inc_closure_${_key}" SET)
    if (_cachedL AND _cachedI)
      get_property(_cachedL GLOBAL PROPERTY "_iface_link_closure_${_key}")
      get_property(_cachedI GLOBAL PROPERTY "_iface_inc_closure_${_key}")
      set(${OUT_L} "${_cachedL}" PARENT_SCOPE)
      set(${OUT_I} "${_cachedI}" PARENT_SCOPE)
      return()
    endif()

    get_property(_resolving GLOBAL PROPERTY "_iface_resolving_${_key}")
    if (_resolving)
      # break cycles
      set(${OUT_L} "" PARENT_SCOPE)
      set(${OUT_I} "" PARENT_SCOPE)
      return()
    endif()
    set_property(GLOBAL PROPERTY "_iface_resolving_${_key}" TRUE)

    # Track this key so the top-level function can clear it later.
    set_property(GLOBAL APPEND PROPERTY _iface_closure_visited_keys "${_key}")

    _get_tprop_list("${_tgt}" INTERFACE_LINK_LIBRARIES _iflibs)
    _get_tprop_list("${_tgt}" INTERFACE_INCLUDE_DIRECTORIES _ifincs)

    if(TARGET "${_tgt}")
      set(_closure_links "${_tgt}")
    else()
      set(_closure_links)
    endif()

    set(_closure_incs "")
    if (_ifincs)
      list(APPEND _closure_incs ${_ifincs})
    endif()
    if (_iflibs)
      list(APPEND _closure_links ${_iflibs})
    endif()

    if (_iflibs)
      foreach(_dep IN LISTS _iflibs)
        if (_dep MATCHES "^\\$<")
          # keep, don't traverse
        elseif (TARGET "${_dep}")
          _resolve_alias("${_dep}" _realdep)

          # Check the frontier predicate before recursing.
          if (NOT "${_stop_func}" STREQUAL "")
            _check_frontier("${_realdep}" "${_stop_func}" _on_frontier)
            if (_on_frontier)
              if (NOT _incl_frontier)
                list(REMOVE_ITEM _closure_links "${_dep}")
              endif()
              continue()
            endif()
          endif()

          _collect_target("${_realdep}" "${_stop_func}" "${_incl_frontier}"
                          _dep_links _dep_incs)
          if (_dep_links)
            list(APPEND _closure_links ${_dep_links})
          endif()
          if (_dep_incs)
            list(APPEND _closure_incs  ${_dep_incs})
          endif()
        else()
          # non-target literal; already included
        endif()
      endforeach()
    endif()

    if (_closure_links)
      list(REMOVE_DUPLICATES _closure_links)
    endif()
    if (_closure_incs)
      list(REMOVE_DUPLICATES _closure_incs)
    endif()

    set_property(GLOBAL PROPERTY "_iface_link_closure_${_key}" "${_closure_links}")
    set_property(GLOBAL PROPERTY "_iface_inc_closure_${_key}"  "${_closure_incs}")
    set_property(GLOBAL PROPERTY "_iface_resolving_${_key}" FALSE)

    set(${OUT_L} "${_closure_links}" PARENT_SCOPE)
    set(${OUT_I} "${_closure_incs}"  PARENT_SCOPE)
  endfunction()

  # Clear the visited-keys list before starting.
  set_property(GLOBAL PROPERTY _iface_closure_visited_keys "")

  set(_all_links "")
  set(_all_incs  "")

  foreach(_item IN LISTS _roots)
    if (_item MATCHES "^\\$<")
      list(APPEND _all_links "${_item}")
      continue()
    endif()
    if (NOT TARGET "${_item}")
      if (NOT "${_item}" STREQUAL "")
        list(APPEND _all_links "${_item}")
      endif()
      continue()
    endif()
    _resolve_alias("${_item}" _t)
    _collect_target("${_t}" "${_stop_at_func}" "${_include_frontier}" _L _I)
    if (_L)
      list(APPEND _all_links ${_L})
    endif()
    if (_I)
      list(APPEND _all_incs  ${_I})
    endif()
  endforeach()

  # Clear all memoized values so successive top-level calls (possibly with
  # different STOP_AT / INCLUDE_FRONTIER options) start with a clean slate.
  get_property(_visited GLOBAL PROPERTY _iface_closure_visited_keys)
  foreach(_key IN LISTS _visited)
    set_property(GLOBAL PROPERTY "_iface_link_closure_${_key}")
    set_property(GLOBAL PROPERTY "_iface_inc_closure_${_key}")
    set_property(GLOBAL PROPERTY "_iface_resolving_${_key}")
  endforeach()
  set_property(GLOBAL PROPERTY _iface_closure_visited_keys)

  if (_all_links)
    list(REMOVE_DUPLICATES _all_links)
  endif()
  if (_all_incs)
    list(REMOVE_DUPLICATES _all_incs)
  endif()

  set(${OUT_LINKS} "${_all_links}" PARENT_SCOPE)
  set(${OUT_INCS}  "${_all_incs}"  PARENT_SCOPE)
endfunction()
