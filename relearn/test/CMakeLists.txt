enable_testing()

# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
        "${proc_count}"
        CACHE STRING "number of jobs that run the tests")


# # googletest# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.18.0
        FIND_PACKAGE_ARGS NAMES GTest)

# google test
set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Neuron models has an additional shared lib that is added via EXTRA_LIBS
add_executable(relearn_tests)
target_sources(
        relearn_tests
        PRIVATE RelearnTest.cpp
        activity_input/test_activity_input_parser.cpp
        activity_input/test_combined_activity_input.cpp
        activity_input/test_constant_activity_input.cpp
        activity_input/test_fast_normal_activity_input.cpp
        activity_input/test_flexible_activity_input.cpp
        activity_input/test_normal_activity_input.cpp
        activity_input/test_scale_activity_input.cpp
        activity_input/test_synaptic_equally_weighted_activity_input.cpp
        activity_input/test_synaptic_scaling_activity_input.cpp
        activity_input/test_synaptic_individually_weighted_activity_input.cpp
        axons/test_axons.cpp
        barnes_hut/test_barnes_hut.cpp
        barnes_hut/test_barnes_hut_base.cpp
        barnes_hut/test_barnes_hut_base_inverted.cpp
        barnes_hut/test_barnes_hut_inverted.cpp
        calcium_calculator/test_calcium_calculator.cpp
        cell/test_cell.cpp
        combined_algorithms/test_combined_algorithms.cpp
        connector/test_connector.cpp
        element_base/test_element_base.cpp
        event/test_event.cpp
        fast_multipole_method/test_fast_multipole_method.cpp
        fired_status_communicator/test_fired_status_approximator.cpp
        fired_status_communicator/test_fired_status_communication_map.cpp
        fired_status_communicator/test_fire_status_communicator_gpu.cpp
        fired_status_recorder/test_fired_status_recorder.cpp
        helper/test_distant_neuron_request.cpp
        helper/test_rank_neuron_id.cpp
        helper/test_synapse_creation_request.cpp
        helper/test_synapse_deletion_request.cpp
        interactive_neuron_io/test_interactive_neuron_io.cpp
        kernel/test_kernel.cpp
        kernel/test_kernel_gamma.cpp
        kernel/test_kernel_gaussian.cpp
        kernel/test_kernel_linear.cpp
        kernel/test_kernel_weibull.cpp
        local_group_translator/test_local_group_translator.cpp
        memory_holder/test_memory_holder.cpp
        misc/test_misc.cpp
        network_graph/test_network_graph.cpp
        network_graph/test_network_graph_gpu.cpp
        neuron_assignment/test_neuron_assignment.cpp
        neuron_extra_info/test_neuron_extra_info.cpp
        neuron_id/test_neuron_id.cpp
        neuron_io/test_neuron_io.cpp
        neuron_io/test_background_activity_io.cpp
        neuron_io/test_synaptic_elements_io.cpp
        neuron_io/test_neuron_to_algorithm_io.cpp
        neuron_models/test_aeif_model.cpp
        neuron_models/test_fitzhughnagumo_model.cpp
        neuron_models/test_izhikevich_model.cpp
        neuron_models/test_poisson_model.cpp
        neuron_models/test_neuron_models.cpp
        neuron_monitor/test_neuron_monitor.cpp
        neurons/test_neurons.cpp
        neurons/test_neurons.cpp
        octree/test_octree.cpp
        octree_node/test_octree_node.cpp
        parser/test_interval_parser.cpp
        parser/test_monitor_parser.cpp
        parser/test_neuronid_parser.cpp
        parser/test_step_parser.cpp
        partition/test_partition.cpp
        probability_picker/test_probability_picker.cpp
        random/test_random.cpp
        relearn_allocator/test_relearn_allocator.cpp
        relearn_exception/test_relearn_exception.cpp
        statistical_measure/test_statistical_measure.cpp
        synapse_deletion_finder/test_synapse_deletion_finder.cpp
        timers/test_timers.cpp
        barnes_hut_cuda/test_barnes_hut_cuda.cpp
        naive_cuda/test_naive_cuda.cpp
        memory_pool/test_memory_pool.cpp
        random/test_random_gpu.cpp
        bloom_filter/test_bloom_filter.cpp
        gpu_cpu_parity/test_gaussian_kernel_parity.cpp
        calcium_gpu/test_calcium_gpu.cpp
        neuron_models_gpu/test_neuron_models_gpu.cpp
        activity_input_gpu/test_activity_input_gpu.cpp
        activity_input_gpu/test_synaptic_equally_weighted_launch.cpp
        mpi_cuda/test_mpi_cuda.cpp
        exchange_algorithm/test_exchange_algorithm.cpp
        neuron_monitor_gpu/test_neuron_monitor_gpu.cpp
        # for visual studio
        ${relearn_tests_additional_files})
target_sources(
        relearn_tests
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/RelearnTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_activity_input_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_combined_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_constant_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_fast_normal_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_flexible_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_normal_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_scale_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_synaptic_equally_weighted_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_synaptic_scaling_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/activity_input/test_synaptic_individually_weighted_activity_input.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/axons/test_axons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/barnes_hut/test_barnes_hut.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/barnes_hut/test_barnes_hut_base.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/barnes_hut/test_barnes_hut_base_inverted.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/barnes_hut/test_barnes_hut_inverted.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/calcium_calculator/test_calcium_calculator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cell/test_cell.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/combined_algorithms/test_combined_algorithms.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/connector/test_connector.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/element_base/test_element_base.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/event/test_event.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/fast_multipole_method/test_fast_multipole_method.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/fired_status_communicator/test_fired_status_approximator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/fired_status_communicator/test_fired_status_communication_map.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/fired_status_recorder/test_fired_status_recorder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/helper/test_distant_neuron_request.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/helper/test_rank_neuron_id.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/helper/test_synapse_creation_request.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/helper/test_synapse_deletion_request.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/interactive_neuron_io/test_interactive_neuron_io.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_kernel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_kernel_gamma.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_kernel_gaussian.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_kernel_linear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_kernel_weibull.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/local_group_translator/test_local_group_translator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory_holder/test_memory_holder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/misc/test_misc.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/test_network_graph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/test_network_graph_gpu.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_assignment/test_neuron_assignment.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_extra_info/test_neuron_extra_info.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_id/test_neuron_id.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_io/test_neuron_io.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_io/test_background_activity_io.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_io/test_synaptic_elements_io.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_io/test_neuron_to_algorithm_io.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_models/test_aeif_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_models/test_fitzhughnagumo_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_models/test_izhikevich_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_models/test_poisson_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_models/test_neuron_models.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_monitor/test_neuron_monitor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neurons/test_neurons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/neurons/test_neurons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/octree/test_octree.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/octree_node/test_octree_node.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parser/test_interval_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parser/test_monitor_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parser/test_neuronid_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parser/test_step_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/partition/test_partition.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/probability_picker/test_probability_picker.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/random/test_random.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/relearn_allocator/test_relearn_allocator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/relearn_exception/test_relearn_exception.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/statistical_measure/test_statistical_measure.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/synapse_deletion_finder/test_synapse_deletion_finder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/timers/test_timers.cpp"
)

target_include_directories(relearn_tests PRIVATE "${PROJECT_SOURCE_DIR}/harness")
target_include_directories(relearn_tests PRIVATE "${PROJECT_SOURCE_DIR}/source" "${PROJECT_SOURCE_DIR}/source/cuda")
target_include_directories(relearn_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

set(RELEARN_TEST_VERIFY_TARGETS "relearn_tests")

target_link_libraries(
        relearn_tests
        PUBLIC project_options
        project_libraries
        relearn_lib
        relearn_harness
        GTest::gtest)
set_target_properties(relearn_tests PROPERTIES CXX_STANDARD 20 ENABLE_EXPORTS 1)


get_target_property(relearn_tests_sources relearn_tests SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_sources})
get_target_property(relearn_tests_headers relearn_tests HEADERS)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_headers})

if (RELEARN_CUDA_ENABLED)
    add_library(relearn_tests_gpu STATIC)
    target_sources(
            relearn_tests_gpu
            PRIVATE
            network_graph/test_network_graph_gpu_device.cu
            memory_pool/test_memory_pool.cu
            random/test_random_gpu.cu
            bloom_filter/test_bloom_filter.cu
            gpu_cpu_parity/test_gaussian_kernel_parity.cu
            neuron_monitor_gpu/test_neuron_monitor_gpu.cu
    )
    target_include_directories(relearn_tests_gpu PRIVATE "${CMAKE_SOURCE_DIR}/source")
    target_include_directories(relearn_tests_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(relearn_tests_gpu PRIVATE cuda_std_20)
    set_target_properties(relearn_tests_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CXX_STANDARD 20
    )
    target_link_libraries(relearn_tests_gpu PUBLIC relearn_lib)
    target_link_libraries(
            relearn_tests
            PUBLIC relearn_tests_gpu)

    get_target_property(relearn_tests_gpu_sources relearn_tests_gpu SOURCES)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_gpu_sources})
    get_target_property(relearn_tests_gpu_headers relearn_tests_gpu HEADERS)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_gpu_headers})
endif ()
