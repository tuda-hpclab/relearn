# Coverage report
#
# Instrumenting is opt-in, not a property of the build type: `--coverage` costs
# compile and run time in every translation unit, and the gcov counters are
# incremented without synchronization, which ThreadSanitizer reports as a data
# race of the instrumentation's own making. Only the configurations whose CI job
# publishes a coverage report ask for it, see cicd/build-and-test.yml; every
# other Debug build - the sanitizer and the checker builds above all - stays
# uninstrumented.
option(COLLECT_COVERAGE "Instrument this build with gcov counters" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND COLLECT_COVERAGE)
  message("BuildType is debug and COLLECT_COVERAGE is on. Will collect coverage information")
  add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:--coverage>)
  # nvcc does not know --coverage, it only knows how to hand a flag to the host
  # compiler it calls for the host part of a .cu file.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--coverage>)
  # nvcc deletes its intermediate host-code stub (.../tmpxft_*.cudafe1.stub.c)
  # after compiling it, but that stub is exactly the file gcov has to open to
  # report the host code of a .cu file. --keep-dir keeps it at a stable path
  # instead of leaving gcov with a name that no longer exists.
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nvcc_keep)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${CMAKE_BINARY_DIR}/nvcc_keep>)
  # Static libraries ignore link options, so the device link of relearn_gpu does
  # not see this and nvcc is never handed a flag it does not know.
  add_link_options(--coverage)
else()
  message("Not a coverage build (${CMAKE_BUILD_TYPE}, COLLECT_COVERAGE=${COLLECT_COVERAGE}). Will not collect coverage")
endif()

# Verify that all files contained in the directory passed as a `DIR` argument
# are part of one of the targets passed in `TARGETS`. This function will warn
# for each file it can not detect is part of one of the passed targets.
function(verify_source_files_in_targets)
  set(oneValueArgs DIR)
  set(multiValueArgs TARGETS IGNORE)
  cmake_parse_arguments(
    VERIFY
    ""
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})

  # Files listed here are conditionally attached to a target depending on build options
  # (e.g. *CPU.cpp sources only added to relearn_lib in the RELEARN_CUDA_ENABLED else()
  # branch, see source/CMakeLists.txt) -- resolve them the same way as target sources so
  # they don't look orphaned in the configurations where they're legitimately excluded.
  set(ignored_files "")
  foreach(file ${VERIFY_IGNORE})
    file(REAL_PATH ${file} file BASE_DIRECTORY ${VERIFY_DIR})
    list(APPEND ignored_files ${file})
  endforeach()

  set(files "")
  foreach(target ${VERIFY_TARGETS})
    # Sources/headers may have been added as paths relative to the target's OWN
    # CMakeLists.txt directory (e.g. relearn_lib's cuda/*.cpp entries, added from
    # source/CMakeLists.txt), not relative to whichever directory calls this function
    # (e.g. source/cuda/CMakeLists.txt). Resolve relative to each target's SOURCE_DIR
    # instead of the caller's CMAKE_CURRENT_SOURCE_DIR, or cross-directory targets
    # silently never match (their files then look "without an associated target").
    get_target_property(verify_source_dir ${target} SOURCE_DIR)
    get_target_property(verify_sources ${target} SOURCES)
    get_target_property(verify_headers ${target} HEADER_SET)
    set(files_tmp "")
    if(verify_sources)
      list(APPEND files_tmp ${verify_sources})
    endif()
    if(verify_headers)
      list(APPEND files_tmp ${verify_headers})
    endif()
    foreach(file ${files_tmp})
      file(REAL_PATH ${file} file BASE_DIRECTORY ${verify_source_dir})
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
  # .cu/.cuh sources are only ever attached to a target when CUDA is enabled -- in a
  # CPU-only configure they're legitimately unreferenced, not orphaned, so skip them here.
  if(RELEARN_CUDA_ENABLED)
    file(GLOB_RECURSE files_in_tree_tmp "*.cu")
    list(APPEND files_in_tree ${files_in_tree_tmp})
    file(GLOB_RECURSE files_in_tree_tmp "*.cuh")
    list(APPEND files_in_tree ${files_in_tree_tmp})
  endif()

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

  if (ignored_files)
    list(REMOVE_ITEM files_in_tree ${ignored_files})
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
