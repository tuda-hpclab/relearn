/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutCUDA.h"

#include "Config.h"

#include "algorithm/Connector.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/util/Util.h"
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
#endif
#include "algorithm/Kernel/Gaussian.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaTypes.h"
#include "cuda/random/RandomNumberHost.h"
#include "cuda/spikes/ExchangeAlgorithm.h"
#include "io/LogFiles.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

void BarnesHutCUDA::init([[maybe_unused]] const number_neurons_type number_neurons) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    OctreeAlgorithm::init(number_neurons);

    const auto gaussian_kernel = std::dynamic_pointer_cast<GaussianDistributionKernel>(kernel);
    RelearnException::check(gaussian_kernel != nullptr, "BarnesHutCUDA::init: Cuda supports only the Gaussian kernel");
    const auto sigma = gaussian_kernel->get_sigma();
    RelearnException::check(sigma > 0, "BarnesHutCUDA::init: Sigma must be > 0");
    squared_sigma_inv = CudaConfig::gaussian_type{ 1 } / (sigma * sigma);
#endif
}

void BarnesHutCUDA::update_remote_nodes() {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    if (number_ranks == 1) {
        return;
    }

    const auto& local_branch_nodes = linearized_tree.get_local_branch_nodes();

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    BarnesHutCUDA_CU::update_remote_nodes_host(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()), number_ranks, local_branch_nodes,
                                               population_ex_storage.handle(), tree_storage.neuron_ids->device_ptr(), my_rank.get_rank(), exc_stream);

    BarnesHutCUDA_CU::update_remote_nodes_host(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()), number_ranks, local_branch_nodes,
                                               population_inh_storage.handle(), tree_storage.neuron_ids->device_ptr(), my_rank.get_rank(), inh_stream);

#endif
}

void BarnesHutCUDA::init_octree() {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else

    linearized_tree = LinearizedTree(get_octree(), synaptic_elements);

    // get_octree()->print_to_file(fmt::format("/home/marvinkaster/ClionProjects/relearn/relearn/octree_{}.txt", mpiPP::MPIInfo::get_my_rank_str()));

    // copy general information of linearized tree on the GPU
    auto [new_tree_storage, mem_usage_1] = BarnesHutCUDA_CU::init_neurons(
        linearized_tree.get_child_indices(), linearized_tree.get_parent_indices(),
        linearized_tree.get_subdomain_lengths(), linearized_tree.get_neuron_ids(),
        linearized_tree.get_rma_offset_to_neuron_id(), linearized_tree.get_node_types());
    tree_storage = std::move(new_tree_storage);

    // copy information for excitatory axons/dendrites
    const auto [h_position_ex, h_axonal_elements_ex, h_dendritic_elements_ex, h_ranks_ex]
        = linearized_tree.get_neuron_details(SignalType::Excitatory);
    auto [new_population_ex_storage, mem_usage_2] = BarnesHutCUDA_CU::init_neuron_details(
        h_position_ex, h_axonal_elements_ex, h_dendritic_elements_ex, h_ranks_ex);
    population_ex_storage = std::move(new_population_ex_storage);

    // copy information for inhibitory axons/dendrites
    const auto [h_position_inh, h_axonal_elements_inh, h_dendritic_elements_inh, h_ranks_inh]
        = linearized_tree.get_neuron_details(SignalType::Inhibitory);
    auto [new_population_inh_storage, mem_usage_3] = BarnesHutCUDA_CU::init_neuron_details(
        h_position_inh, h_axonal_elements_inh, h_dendritic_elements_inh, h_ranks_inh);
    population_inh_storage = std::move(new_population_inh_storage);

    total_mem_usage = mem_usage_1 + mem_usage_2 + mem_usage_3;

    linearized_tree_initialized = true;
#endif
}

void BarnesHutCUDA::free() {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    tree_storage = {};
    population_ex_storage = {};
    population_inh_storage = {};
    vacant_exc_axons_mapping = {};
    vacant_inh_axons_mapping = {};
#endif
}

void BarnesHutCUDA::set_acceptance_criterion([[maybe_unused]] const acceptance_criterion_type new_acceptance_criterion) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    RelearnException::check(new_acceptance_criterion > acceptance_criterion_type{ 0 },
                            "BarnesHutCUDA::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})",
                            new_acceptance_criterion);
    acceptance_criterion = new_acceptance_criterion;
#endif
}

void BarnesHutCUDA::update_linearized_tree() {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    // linearized_tree.update_linearized_tree(synaptic_elements);
#endif
}

void BarnesHutCUDA::prepare_update_connectivity(const std::span<const SignalType> signal_types,
                                                const std::span<const counter_type> vacant_axons,
                                                const std::span<const counter_type> vacant_excitatory_dendrites,
                                                const std::span<const counter_type> vacant_inhibitory_dendrites) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else

    if (!linearized_tree_initialized) {
        OctreeAlgorithm::update_tree(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
        init_octree();
        LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Linearized octree created");
    }

    Timers::start(TimerRegion::UPDATE_LEAF_NODES);

    // CUDA update leaf nodes on GPU
    const auto axon_handle_const = synaptic_elements->get_axons()->get_cuda_handle_const();
    const auto dend_exc_handle_const = synaptic_elements->get_dendrites()->get_cuda_handle_const(SignalType::Excitatory);
    const auto dend_inh_handle_const = synaptic_elements->get_dendrites()->get_cuda_handle_const(SignalType::Inhibitory);
    const auto* d_signal_types = synaptic_elements->get_axons()->get_d_signal_types();

    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));

    BarnesHutCUDA_CU::update_leaf_nodes_entry(tree, d_signal_types, axon_handle_const, dend_exc_handle_const, dend_inh_handle_const,
                                              TreeVacancyOutputHandle{ population_ex_storage.vacant_axons->device_ptr(), population_ex_storage.vacant_dendrites->device_ptr(), population_inh_storage.vacant_axons->device_ptr(), population_inh_storage.vacant_dendrites->device_ptr() }, exc_stream);

    cudaDeviceSynchronize_bridge();
    cuda_resolve_gpu_timers();

    const auto population_ex = population_ex_storage.handle();
    const auto population_inh = population_inh_storage.handle();

    BarnesHutCUDA_CU::calculate_updated_octree_host(population_ex, tree,
                                                    linearized_tree.get_level_indices(), exc_stream);

    BarnesHutCUDA_CU::calculate_updated_octree_host(population_inh, tree,
                                                    linearized_tree.get_level_indices(), inh_stream);

    update_remote_nodes();

    BarnesHutCUDA_CU::calculate_updated_octree_host(population_ex, tree,
                                                    linearized_tree.get_level_indices(), exc_stream);

    BarnesHutCUDA_CU::calculate_updated_octree_host(population_inh, tree,
                                                    linearized_tree.get_level_indices(), inh_stream);

    // Vacant axon mapping
    // TODO Maybe do this on the gpu?
    auto helper = [this](const CudaConfig::synaptic_count_type* d_vacant_axons_ptr, const std::shared_ptr<StreamWrapper>& stream) {
        std::vector<CudaConfig::synaptic_count_type> h_vacant_axons(linearized_tree.size());
        const auto& h_node_types = linearized_tree.get_node_types();
        cudaMemcpyAsync_to_host_bridge(h_vacant_axons.data(), d_vacant_axons_ptr, linearized_tree.size() * sizeof(CudaConfig::synaptic_count_type), *stream);
        std::vector<CudaConfig::bh_index_type> vacant_axons_mapping{};
        for (auto i = 0U; i < linearized_tree.size(); i++) {
            if (h_node_types[i] == NodeType::Leaf) {
                for (auto j = 0U; j < h_vacant_axons[i]; j++) {
                    vacant_axons_mapping.emplace_back(i);
                }
            }
        }
        return DeviceArray<CudaConfig::bh_index_type>(std::span<const CudaConfig::bh_index_type>(vacant_axons_mapping), stream);
    };

    vacant_exc_axons_mapping = helper(population_ex_storage.vacant_axons->device_ptr(), exc_stream);
    vacant_inh_axons_mapping = helper(population_inh_storage.vacant_axons->device_ptr(), inh_stream);

    cudaDeviceSynchronize_bridge();
    // The device is already caught up above, so resolving the GPU-event timers queued by the four
    // calculate_updated_octree_host() calls above (exc/inh streams, two rounds) is free here -- it
    // never introduces a synchronization point of its own.
    cuda_resolve_gpu_timers();

    Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);
#endif
}

BarnesHutCUDA::target_neuron_both_result_type
BarnesHutCUDA::find_target_neurons([[maybe_unused]] const number_neurons_type number_neurons) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank().get_rank();

    const auto& global_tree = get_octree();

    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linearized_tree.size()));
    const auto population_ex = population_ex_storage.handle();
    const auto population_inh = population_inh_storage.handle();

    auto result_pair = BarnesHutCUDA_CU::find_target_neurons(Config::random_seed, counter++,
                                                             population_ex, vacant_exc_axons_mapping->device_ptr(), vacant_exc_axons_mapping->size(),
                                                             population_inh, vacant_inh_axons_mapping->device_ptr(), vacant_inh_axons_mapping->size(),
                                                             static_cast<CudaConfig::number_neurons_type>(number_neurons), tree,
                                                             Constants::bh_default_theta, population_ex_storage.ranks->device_ptr(),
                                                             my_rank,
                                                             squared_sigma_inv, number_ranks, static_cast<CudaConfig::number_neurons_type>(number_neurons), exc_stream, inh_stream);

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return result_pair;
#endif
}

ProcessRequestsAwareResult BarnesHutCUDA::process_requests(const std::vector<int>& counts_full, const std::vector<int>& offsets_full, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& source_ids, [[maybe_unused]] const DeviceArray<SimpleVec3d>& source_positions, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& target_ids,
                                                           [[maybe_unused]] const SignalType dendrite_type_needed) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    Timers::start(TimerRegion::BLOCK2);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    /*
     * Parse calculation requests
     */

    const auto linear_tree_size = linearized_tree.get_neuron_ids().size();
    const auto rma_size = linearized_tree.get_rma_offset_to_neuron_id().size();
    const auto ranks = linearized_tree.get_neuron_details(dendrite_type_needed).ranks;
    auto d_ranks = DeviceArray<CudaConfig::mpi_rank_type>(ranks);
    auto* const rma_offset_to_index = tree_storage.rma_offset_to_index->device_ptr();
    const auto number_requests = static_cast<std::size_t>(offsets_full[static_cast<std::size_t>(number_ranks) - 1U] + counts_full[static_cast<std::size_t>(number_ranks) - 1]);
    RelearnException::check(target_ids.size() == number_requests, "Wrong {} != {}", target_ids.size(), number_requests);

    const auto d_counts = DeviceArray<int>(counts_full);
    auto source_ranks = DeviceArray<CudaConfig::mpi_rank_type>(number_requests);
    auto d_calculation_responses_cast = DeviceArray<CudaConfig::number_neurons_type>(number_requests);

    Timers::stop_and_add(TimerRegion::BLOCK2);

    const auto tree = tree_storage.handle(static_cast<CudaConfig::bh_index_type>(linear_tree_size));
    const auto population_ex = population_ex_storage.handle();

    process_calculation_requests_entry_aware(Config::random_seed, counter++,
                                             BHCalculationRequestHandle{ d_counts.device_ptr(), source_ids.device_ptr(), source_positions.device_ptr(),
                                                                         target_ids.device_ptr(), d_calculation_responses_cast.device_ptr(), source_ranks.device_ptr(), number_requests },
                                             extra_infos->get_gpu_handle(), population_ex, tree,
                                             Constants::bh_default_theta,
                                             RemoteNodeRankHandle{ d_ranks.device_ptr(), rma_offset_to_index, rma_size },
                                             squared_sigma_inv);

    /*
     * Parse creation requests
     */

    Timers::start(TimerRegion::BLOCK1);

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty, "ForwardConnector::process_requests: The synaptic elements are empty");

    auto d_flat_responses = DeviceArray<SynapseCreationResponse>(number_requests);

    const auto den_exc_handle = synaptic_elements->get_dendrites()->get_cuda_handle(SignalType::Excitatory);
    const auto den_inh_handle = synaptic_elements->get_dendrites()->get_cuda_handle(SignalType::Inhibitory);

    auto created_synapses = process_requests_entry_aware(
        SynapseCreationRequestHandle{ source_ids.device_ptr(), source_ranks.device_ptr(), d_calculation_responses_cast.device_ptr(), dendrite_type_needed, number_requests },
        d_flat_responses.device_ptr(), den_exc_handle, den_inh_handle, extra_infos->get_gpu_handle(),
        network_graph->get_gpu_handle(), Config::random_seed, counter++);

    // den_exc_handle/den_inh_handle above already requested non-const device pointers via
    // get_cuda_handle(), which already marked grown/delta/vacant/connected device-modified.
    network_graph->device_was_modified();

    Timers::stop_and_add(TimerRegion::BLOCK1);

    return { std::move(d_flat_responses), std::move(d_calculation_responses_cast), created_synapses };
#endif
}

void BarnesHutCUDA::process_responses([[maybe_unused]] const DeviceArray<SynapseCreationResponse>& responses, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& source_ids, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& target_ids, const std::vector<int>& sizes, const std::vector<int>& offset, [[maybe_unused]] const SignalType signal_type) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    Timers::start(TimerRegion::BLOCK2);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();
    const auto total_number_requests = static_cast<std::size_t>(offset[number_ranks - 1] + sizes[number_ranks - 1]);

    const auto synaptic_elements_empty = synaptic_elements != nullptr;
    RelearnException::check(synaptic_elements_empty, "ForwardConnector::process_responses: The synaptic elements are empty");

    const auto axon_handle = synaptic_elements->get_axons()->get_cuda_handle();

    const DeviceArray<CudaConfig::mpi_rank_type> target_ranks(static_cast<std::size_t>(total_number_requests));
    const DeviceArray<int> d_sizes(sizes);
    process_responses_entry_aware(d_sizes.device_ptr(), SynapseCreationResponseHandle{ source_ids.device_ptr(), target_ids.device_ptr(), signal_type, total_number_requests },
                                  responses.device_ptr(), axon_handle, extra_infos->get_gpu_handle(),
                                  network_graph->get_gpu_handle(), target_ranks.device_ptr());

    cudaDeviceSynchronize_bridge();
    Timers::stop_and_add(TimerRegion::BLOCK2);

    Timers::start(TimerRegion::BLOCK3);
    Timers::start(TimerRegion::BLOCK4);
    network_graph->device_was_modified();
    Timers::stop_and_add(TimerRegion::BLOCK4);

    // axon_handle above already requested a non-const device pointer via get_cuda_handle(), which
    // already marked grown/delta/vacant/connected device-modified.
    Timers::stop_and_add(TimerRegion::BLOCK3);

#endif
}

#pragma GCC diagnostic pop