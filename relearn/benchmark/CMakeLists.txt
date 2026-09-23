# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(BENCHMARK_ENABLE_TESTING
        OFF
        CACHE INTERNAL "")
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
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
        benchmark_barnes_hut_cuda.cpp
        benchmark_calcium.cpp
        benchmark_exchange_algorithm_cuda.cpp
        benchmark_kernel.cpp
        benchmark_memory_pool.cpp
        benchmark_naive.cpp
        benchmark_naive_cuda.cpp
        benchmark_network_graph.cpp
        benchmark_network_graph_gpu_edge_iteration.cpp
        benchmark_neuron_model.cpp
        benchmark_preparing_fire_exchange.cpp
        benchmark_octree.cpp
        benchmark_synaptic_elements.cpp
        benchmark_synaptic_input_cuda.cpp
        benchmark_synaptic_input_prepare_cuda.cpp
)

target_sources(
        relearn_benchmarks
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/main.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/util.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_network_graph_gpu_edge_iteration.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_memory_pool.h")

target_include_directories(relearn_benchmarks PRIVATE "${PROJECT_SOURCE_DIR}/harness")
target_include_directories(relearn_benchmarks PRIVATE "${PROJECT_SOURCE_DIR}/source" "${PROJECT_SOURCE_DIR}/source/cuda")
target_include_directories(relearn_benchmarks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(relearn_benchmarks PRIVATE project_options
        project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib relearn_harness)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)
set_target_properties(relearn_benchmarks PROPERTIES CXX_STANDARD 20)

get_target_property(relearn_benchmark_sources relearn_benchmarks SOURCES)
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${relearn_benchmark_sources})

if (RELEARN_CUDA_ENABLED)
    add_library(relearn_benchmarks_gpu STATIC)
    target_sources(
            relearn_benchmarks_gpu
            PRIVATE
            benchmark_network_graph_gpu_edge_iteration_kernels.cu
            benchmark_memory_pool_kernels.cu
    )
    target_include_directories(relearn_benchmarks_gpu PRIVATE "${CMAKE_SOURCE_DIR}/source")
    target_include_directories(relearn_benchmarks_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(relearn_benchmarks_gpu PRIVATE cuda_std_20)
    set_target_properties(relearn_benchmarks_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CXX_STANDARD 20
    )
    target_link_libraries(relearn_benchmarks_gpu PUBLIC relearn_lib)
    target_link_libraries(
            relearn_benchmarks
            PUBLIC relearn_benchmarks_gpu)
endif ()

set(relearn_benchmarks_verify_targets relearn_benchmarks)
if (RELEARN_CUDA_ENABLED)
    list(APPEND relearn_benchmarks_verify_targets relearn_benchmarks_gpu)
endif ()

verify_source_files_in_targets(
        TARGETS
        ${relearn_benchmarks_verify_targets}
        DIR
        ${CMAKE_CURRENT_SOURCE_DIR})
