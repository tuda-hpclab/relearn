cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

project(relearn)
enable_language(CXX)

include(CheckLanguage)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

# project options
add_library(project_options INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# Add Cuda support
include(cmake/Cuda.cmake)

# Choose the floating point precision of the simulation
include(cmake/Precision.cmake)

# Add linker configuration
include(cmake/Linker.cmake)
configure_linker(project_options)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# instrumentation support
include(cmake/Instrumentation.cmake)

# make libraries available
include(cmake/ProjectLibraries.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# utilities
include(cmake/Utils.cmake)

add_subdirectory(source)

if(NOT ENABLE_SCOREP)
  # Must run at this (root) scope for ctest to generate a top-level CTestTestfile.cmake that
  # recurses into every add_subdirectory() below -- test/CMakeLists.txt's own enable_testing()
  # call only covers that directory's scope, not sibling ones like test-mpi.
  enable_testing()
  add_subdirectory(benchmark)
  add_subdirectory(harness)
  add_subdirectory(test)
endif()
