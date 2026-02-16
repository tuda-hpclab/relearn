enable_testing()

# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
    "${proc_count}"
    CACHE STRING "number of jobs that run the tests")

set(check_args -j${CTEST_JOB_COUNT} --output-on-failure ${CTEST_EXTRA_ARGS})

add_custom_target(
  check
  COMMAND
    ${CMAKE_COMMAND} -E env
    $<$<BOOL:${ENABLE_SANITIZER_ADDRESS}>:ASAN_OPTIONS=fast_unwind_on_malloc=0,symbolize=1>
    $<$<BOOL:${ENABLE_SANITIZER_LEAK}>:LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/cmake/sanitizers/lsan.supp>
    ${CMAKE_CTEST_COMMAND} ${check_args}
  USES_TERMINAL)

if(${COVERAGE_REPORT_AVAILABLE})
  add_custom_target(
    check-coverage
    DEPENDS check
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
            coverage-report)
endif()

# # googletest# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest)

# google test
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(gtest_main_includes gtest_main
                    INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                            "${gtest_main_includes}")

# The base library for all tests. Is automatically linked to in the
# add_relearn_test function call.

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
          bounding_box/test_bounding_box.cpp
          calcium_calculator/test_calcium_calculator.cpp
          cell/test_cell.cpp
          combined_algorithms/test_combined_algorithms.cpp
          connector/test_connector.cpp
          element_base/test_element_base.cpp
          event/test_event.cpp
          fast_multipole_method/test_fast_multipole_method.cpp
          fired_status_communicator/test_fired_status_approximator.cpp
          fired_status_communicator/test_fired_status_communication_map.cpp
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
          node_cache/test_node_cache.cpp
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
          space_filling_curve/test_hilbert.cpp
          space_filling_curve/test_morton.cpp
          statistical_measure/test_statistical_measure.cpp
          string_util/test_string_util.cpp
          synapse_deletion_finder/test_synapse_deletion_finder.cpp
          timers/test_timers.cpp
          util/test_util.cpp
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
     "${CMAKE_CURRENT_SOURCE_DIR}/bounding_box/test_bounding_box.cpp"
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
     "${CMAKE_CURRENT_SOURCE_DIR}/node_cache/test_node_cache.cpp"
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
     "${CMAKE_CURRENT_SOURCE_DIR}/space_filling_curve/test_hilbert.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/space_filling_curve/test_morton.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/statistical_measure/test_statistical_measure.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/string_util/test_string_util.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/synapse_deletion_finder/test_synapse_deletion_finder.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/timers/test_timers.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/util/test_util.cpp"
)

target_include_directories(relearn_tests PRIVATE "${CMAKE_SOURCE_DIR}/harness")
target_include_directories(relearn_tests PRIVATE "${CMAKE_SOURCE_DIR}/source")
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

add_test(NAME SerialTests COMMAND relearn_tests)
add_dependencies(check relearn_tests)

get_target_property(relearn_tests_sources relearn_tests SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_sources})
get_target_property(relearn_tests_headers relearn_tests HEADERS)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_tests_headers})
