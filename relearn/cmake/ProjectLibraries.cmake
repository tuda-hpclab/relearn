# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

include(cmake/CPM.cmake)

add_library(project_libraries INTERFACE)
add_library(project_libraries_gpu INTERFACE)

include(FetchContent)

find_package(Threads REQUIRED)
target_link_libraries(project_libraries INTERFACE Threads::Threads)

if(WIN32)
  add_compile_options(
	  "$<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>"
	)
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(project_options INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

if(UNIX)
  target_link_libraries(project_options INTERFACE stdc++fs)
  target_link_libraries(project_libraries INTERFACE stdc++fs)
endif()

option(ENABLE_MPI "Enable mpi" ON)
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_CXX_FOUND)
    target_compile_definitions(project_options
                               INTERFACE -DMPI_FOUND=$<BOOL:${MPI_CXX_FOUND}>)

    # fix CI build issue
    get_target_property(mpi_cxx_compile_options MPI::MPI_CXX
                        INTERFACE_COMPILE_OPTIONS)

    if("${mpi_cxx_compile_options}" MATCHES "-flto=auto")
      message(
        WARNING
          "MPI_CXX was compiled with -flto=auto and -ffat-lto-objects, removing lto flags to prevent CI build failure"
      )
      string(
        REPLACE "-flto=auto"
                ""
                mpi_cxx_compile_options
                ${mpi_cxx_compile_options})
      string(
        REPLACE "-ffat-lto-objects"
                ""
                mpi_cxx_compile_options
                ${mpi_cxx_compile_options})
    endif()

    set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS
                                                  "${mpi_cxx_compile_options}")

    # Some MPI installations (e.g. OpenMPI with the legacy mpicxx bindings enabled) ship headers
    # that don't pass our warning flags (old-style casts, etc.). Mark them SYSTEM, same as the
    # other third-party dependencies below, so warnings originating there don't fail the build.
    foreach(mpi_target MPI::MPI_C MPI::MPI_CXX)
      if(TARGET ${mpi_target})
        get_target_property(mpi_includes ${mpi_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(mpi_includes)
          set_target_properties(${mpi_target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                                          "${mpi_includes}")
        endif()
      endif()
    endforeach()

    target_link_libraries(project_libraries INTERFACE MPI::MPI_CXX)
  endif()
endif()

include_directories("external/")
set(CPP_UTILITIES_ENABLE_TESTING OFF)
set(MPI_WRAPPER_ENABLE_TESTING OFF)

if(WIN32)
  target_compile_definitions(project_options
                               INTERFACE -DBOOST_ALL_NO_LIB)
else()
  set(BOOST_ENABLE_CMAKE ON)
  CPMAddPackage(
          NAME Boost
          VERSION 1.89.0
          URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
          URL_HASH SHA256=67acec02d0d118b5de9eb441f5fb707b3a1cdd884be00ca24b9a73c995511f74
          OPTIONS "BOOST_ENABLE_CMAKE ON"
  )

  target_link_libraries(project_options INTERFACE Boost::random Boost::json Boost::lexical_cast)

  # Boost is a vendored dependency; mark its headers SYSTEM like the other fetched
  # dependencies below so warnings triggered by template instantiation in our TUs
  # (e.g. inside boost::lexical_cast) don't surface as our own warnings.
  foreach(boost_target IN ITEMS boost_random boost_json boost_lexical_cast)
    if(TARGET ${boost_target})
      get_target_property(boost_target_includes ${boost_target} INTERFACE_INCLUDE_DIRECTORIES)
      if(boost_target_includes)
        set_target_properties(${boost_target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                                          "${boost_target_includes}")
      endif()
    endif()
  endforeach()

  # BOOST_ENABLE_CMAKE ON configures every library in the Boost superbuild (locale, wave,
  # serialization, ...), not just the three we link against, so their own .cpp sources get
  # compiled -- and warn -- regardless of what we use. We don't control that vendored code,
  # so recursively strip warnings from every target Boost's own build defines, whichever
  # compiler/version next trips over one of them.
  function(relearn_disable_warnings_recursive dir)
    get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
      get_target_property(target_type ${target} TYPE)
      if(target_type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
        target_compile_options(${target} PRIVATE -w)
      endif()
    endforeach()
    get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
      relearn_disable_warnings_recursive("${subdir}")
    endforeach()
  endfunction()

  if(Boost_SOURCE_DIR)
    relearn_disable_warnings_recursive("${Boost_SOURCE_DIR}")
  endif()

endif()

if(ENABLE_CUDA)
  CPMAddPackage(
          NAME cuco
          GITHUB_REPOSITORY NVIDIA/cuCollections
          GIT_TAG dev
          OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_BENCHMARKS OFF"
          "BUILD_EXAMPLES OFF"
  )

  target_link_libraries(project_libraries_gpu INTERFACE cuco)

endif()

# declaration
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
cmake_policy(SET CMP0077 NEW)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 12.2.0
  FIND_PACKAGE_ARGS NAMES fmt)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.17.0)

# range-v3
FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3
  GIT_TAG 7e6f34b1e820fb8321346888ef0558a0ec842b8e)

# ctpg
FetchContent_Declare(
  ctpg
  GIT_REPOSITORY https://github.com/peter-winter/ctpg
  GIT_TAG v1.3.7)

# fmt
set(FMT_SYSTEM_HEADERS ON)
FetchContent_MakeAvailable(fmt)
target_link_libraries(project_libraries_gpu INTERFACE fmt::fmt)
target_link_libraries(project_libraries INTERFACE fmt::fmt)

# spdlog
set(SPDLOG_FMT_EXTERNAL ON)
FetchContent_MakeAvailable(spdlog)
get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                        "${spdlog_includes}")
# target_link_libraries(project_libraries_gpu INTERFACE spdlog)
target_link_libraries(project_libraries INTERFACE spdlog)

# range-v3
FetchContent_MakeAvailable(range-v3)
get_target_property(range-v3_includes range-v3 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(range-v3 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                          "${range-v3_includes}")
target_link_libraries(project_libraries INTERFACE range-v3)

# ctpg
set(CTPG_ENABLE_TESTS OFF)
FetchContent_MakeAvailable(ctpg)
get_target_property(ctpg_includes ctpg INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(ctpg PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                      "${ctpg_includes}")
target_link_libraries(project_libraries INTERFACE ctpg::ctpg)

# set compile commands back to on
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
