add_library(relearn_harness STATIC)
target_sources(
	relearn_harness
	PRIVATE
			harness.cpp
			
			adapter/connector/ConnectorAdapter.cpp
			adapter/extra_info/ExtraInfoAdapter.cpp
			adapter/fast_multipole_method/FMMAdapter.cpp
			adapter/helper/RankNeuronIdAdapter.cpp
			adapter/interval/IntervalAdapter.cpp
			adapter/neurons/NeuronTypesAdapter.cpp
			adapter/node_cache/NodeCacheAdapter.cpp
			adapter/simulation/SimulationAdapter.cpp
			adapter/synapses/SynapsesAdapter.cpp
			adapter/synaptic_elements/SynapticElementsAdapter.cpp
			
			factory/activity_input/activity_input_factory.cpp
			factory/algorithm/algorithm_config_factory.cpp
			factory/algorithm/algorithm_factory.cpp
			factory/calcium/calcium_factory.cpp
			factory/connector/connector_factory.cpp
			factory/extra_info/extra_info_factory.cpp
			factory/fired_status_communicator/fired_status_communicator_factory.cpp  
			factory/fmm/fmm_factory.cpp
			factory/interval/interval_factory.cpp
			factory/kernel/kernel_factory.cpp
			factory/local_group_translator/local_group_translator_factory.cpp
			factory/mpi_rank/mpi_rank_factory.cpp
			factory/network_graph/network_graph_factory.cpp
			factory/neuron_id/neuron_id_factory.cpp     
			factory/neuron_model/neuron_model_factory.cpp
			factory/neuron_types/neuron_types_factory.cpp     
			factory/neurons/neurons_factory.cpp        
			factory/random/random_factory.cpp
			factory/rank_neuron_id/rank_neuron_id_factory.cpp
			factory/simulation/simulation_factory.cpp	
			factory/synapses/synapses_factory.cpp  
			factory/synaptic_elements/synaptic_elements_factory.cpp
			)

target_sources(
  relearn_harness
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/connector/ConnectorAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/event/EventAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/extra_info/ExtraInfoAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/fast_multipole_method/FMMAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/helper/RankNeuronIdAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/interval/IntervalAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/kernel/KernelAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/mpi/MpiAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/mpi/MpiRankAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/network_graph/NetworkGraphAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/neuron_assignment/NeuronAssignmentAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/neuron_id/NeuronIdAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/neurons/NeuronTypesAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/neurons/NeuronsAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/node_cache/NodeCacheAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/octree/OctreeAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/simulation/SimulationAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/synapses/SynapsesAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/synaptic_elements/SynapticElementsAdapter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/timers/TimersAdapter.h"

    "${CMAKE_CURRENT_SOURCE_DIR}/factory/activity_input/activity_input_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/algorithm/algorithm_config_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/algorithm/algorithm_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/calcium/calcium_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/connector/connector_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/extra_info/extra_info_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/fired_status_communicator/fired_status_communicator_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/fmm/fmm_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/interval/interval_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/kernel/kernel_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/local_group_translator/local_group_translator_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/memory_holder/memory_holder_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/mpi_rank/mpi_rank_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/network_graph/network_graph_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/neuron_id/neuron_id_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/neuron_model/neuron_model_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/neuron_types/neuron_types_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/neurons/neurons_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/octree/octree_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/random/random_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/rank_neuron_id/rank_neuron_id_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/simulation/simulation_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/synapses/synapses_factory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/factory/synaptic_elements/synaptic_elements_factory.h"
)

set_target_properties(relearn_harness PROPERTIES ENABLE_EXPORTS 1)
set_target_properties(relearn_harness PROPERTIES CXX_STANDARD 20)
target_include_directories(relearn_harness PRIVATE "${PROJECT_SOURCE_DIR}/source" "${PROJECT_SOURCE_DIR}/source/cuda")
target_include_directories(relearn_harness PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(relearn_harness PUBLIC project_options project_libraries)
target_link_libraries(relearn_harness PRIVATE relearn_lib)

get_target_property(relearn_harness_sources relearn_harness SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_harness_sources})
get_target_property(relearn_harness_headers relearn_harness HEADERS)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_harness_headers})

verify_source_files_in_targets(TARGETS relearn_harness DIR ${CMAKE_CURRENT_SOURCE_DIR})
