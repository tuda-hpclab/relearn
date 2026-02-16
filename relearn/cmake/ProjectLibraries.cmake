# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

add_library(project_libraries INTERFACE)
add_library(project_libraries_gpu INTERFACE)

include(FetchContent)

find_package(Threads REQUIRED)
target_link_libraries(project_libraries INTERFACE Threads::Threads)

if(WIN32)
  add_compile_options("/openmp:llvm")
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
  find_package(Boost CONFIG REQUIRED COMPONENTS RANDOM JSON)
  # target_link_libraries(project_options INTERFACE Boost::random)

  get_target_property(boost_includes Boost::boost INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(
    Boost::boost PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                            "${boost_includes}")

  target_link_libraries(project_options INTERFACE Boost::random Boost::json)
endif()

# declaration
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
cmake_policy(SET CMP0077 NEW)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 11.0.2
  FIND_PACKAGE_ARGS NAMES fmt)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.15.0)

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
