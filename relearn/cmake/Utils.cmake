# Coverage report
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("BuildType is debug. Will collect coverage information")
  include(cmake/CodeCoverage.cmake)
  append_coverage_compiler_flags()
else()
  message("Not debug. Will not collect coverage $CMAKE_BUILD_TYPE")
endif()

# Verify that all files contained in the directory passed as a `DIR` argument
# are part of one of the targets passed in `TARGETS`. This function will warn
# for each file it can not detect is part of one of the passed targets.
function(verify_source_files_in_targets)
  set(oneValueArgs DIR)
  set(multiValueArgs TARGETS)
  cmake_parse_arguments(
    VERIFY
    ""
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})

  set(files "")
  foreach(target ${VERIFY_TARGETS})
    get_target_property(verify_sources ${target} SOURCES)
    get_target_property(verify_headers ${target} HEADER_SET)
    if(verify_sources)
      list(APPEND files_tmp ${verify_sources})
    endif()
    if(verify_headers)
      list(APPEND files_tmp ${verify_headers})
    endif()
    foreach(file ${files_tmp})
      file(REAL_PATH ${file} file)
      list(APPEND files ${file})
    endforeach()
  endforeach()

  # Find all directories containing CMakeLists.txt
  file(GLOB_RECURSE cmake_dirs RELATIVE ${CMAKE_SOURCE_DIR} "*/CMakeLists.txt")


  # Extract directories and add them to the exclusion list
  foreach(cmake_file ${cmake_dirs})
    get_filename_component(dir ${cmake_file} DIRECTORY)
    list(APPEND excluded_dirs ${dir})
  endforeach()

  # Remove duplicate directories
  list(REMOVE_DUPLICATES excluded_dirs)

  file(GLOB_RECURSE files_in_tree_tmp "*.h")
  list(APPEND files_in_tree ${files_in_tree_tmp})
  file(GLOB_RECURSE files_in_tree_tmp "*.hpp")
  list(APPEND files_in_tree ${files_in_tree_tmp})
  file(GLOB_RECURSE files_in_tree_tmp "*.cpp")
  list(APPEND files_in_tree ${files_in_tree_tmp})
  file(GLOB_RECURSE files_in_tree_tmp "*.cu")
  list(APPEND files_in_tree ${files_in_tree_tmp})
  file(GLOB_RECURSE files_in_tree_tmp "*.cuh")
  list(APPEND files_in_tree ${files_in_tree_tmp})

  # Identify all subdirectories of excluded directories
  foreach(excluded_dir ${excluded_dirs})
    foreach(file ${files_in_tree})
      get_filename_component(file_dir ${file} DIRECTORY)
      string(FIND "${file_dir}" "${excluded_dir}" found_pos)
      if (found_pos EQUAL 0 OR found_pos GREATER -1)
        list(APPEND to_remove ${file})
      endif()
    endforeach()
  endforeach()

  # Remove files that are in excluded directories or their subdirectories
  if (to_remove)
    list(REMOVE_ITEM files_in_tree ${to_remove})
  endif()

  foreach(FILE ${files_in_tree})
    list(
      FIND
      files
      ${FILE}
      RESULT)
    if(${RESULT} EQUAL "-1")
      message(WARNING "Found file without an associated target: ${FILE}.")
    endif()
  endforeach()
endfunction()
