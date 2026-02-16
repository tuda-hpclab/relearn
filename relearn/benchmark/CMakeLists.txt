# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "")
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
  FIND_PACKAGE_ARGS NAMES benchmark)

FetchContent_MakeAvailable(googlebenchmark)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(googlebenchmark_includes benchmark
                    INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(benchmark PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                           "${googlebenchmark_includes}")

add_executable(relearn_benchmarks)
target_sources(
  relearn_benchmarks
  PRIVATE main.cpp
          benchmark_activity_input.cpp
          benchmark_barnes_hut.cpp
          benchmark_calcium.cpp
          benchmark_kernel.cpp
          benchmark_naive.cpp
          benchmark_network_graph.cpp
          benchmark_neuron_model.cpp
          benchmark_octree.cpp
          benchmark_synaptic_elements.cpp
          )

target_sources(
  relearn_benchmarks
  PUBLIC
         "${CMAKE_CURRENT_SOURCE_DIR}/main.h")

target_include_directories(relearn_benchmarks PRIVATE "${CMAKE_SOURCE_DIR}/harness")
target_include_directories(relearn_benchmarks PRIVATE "${CMAKE_SOURCE_DIR}/source")
target_include_directories(relearn_benchmarks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(relearn_benchmarks PRIVATE project_options
                                                 project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib relearn_harness)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)
set_target_properties(relearn_benchmarks PROPERTIES CXX_STANDARD 20)

get_target_property(relearn_benchmark_sources relearn_benchmarks SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_benchmark_sources})

verify_source_files_in_targets(
  TARGETS
  relearn_benchmarks
  DIR
  ${CMAKE_CURRENT_SOURCE_DIR})
