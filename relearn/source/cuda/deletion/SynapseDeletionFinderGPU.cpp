/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinderGPU.h"

#include "Config.h"

#include "cuda/random/RandomNumberHost.h"
#include "cuda/spikes/ExchangeAlgorithm.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cpp-utility/data/partition.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <algorithm>
#include <tuple>

void SynapseDeletionFinderGPU::init(const RelearnTypes::number_neurons_type number_neurons) {
    SynapseDeletionFinderBase::init(number_neurons);

    random_key = RandomNumbers::register_random_numbers(RandomNumberKey::DELETE, RandomNumberType::UNIFORM, number_neurons, Config::random_seed);
}

std::pair<RelearnTypes::number_synapse_type, RelearnTypes::number_synapse_type> SynapseDeletionFinderGPU::delete_synapses() {
    auto deletion_helper_axons = [this](const ElementType element_type, const std::span<const SignalType> signal_types, const DeviceArray<CudaConfig::synaptic_count_type>& to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, std::nullopt, signal_types, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    auto deletion_helper_dendrites = [this](const ElementType element_type, SignalType signal_type, const DeviceArray<CudaConfig::synaptic_count_type>& to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, signal_type, {}, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);
    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto raw_axons = synaptic_elements->commit_updates(SynapticElementType::Axon);
    const DeviceArray<CudaConfig::synaptic_count_type> to_delete_axons{ std::span<const CudaConfig::synaptic_count_type>{ raw_axons } };
    cudaDeviceSynchronize_bridge();
    const auto deleted_axons = deletion_helper_axons(ElementType::Axon, synaptic_elements->get_signal_types(), to_delete_axons);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto raw_exc = synaptic_elements->commit_updates(SynapticElementType::DendriteExcitatory);
    const DeviceArray<CudaConfig::synaptic_count_type> to_delete_excitatory_dendrites{ std::span<const CudaConfig::synaptic_count_type>{ raw_exc } };
    cudaDeviceSynchronize_bridge();
    const auto deleted_excitatory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Excitatory, to_delete_excitatory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto raw_inh = synaptic_elements->commit_updates(SynapticElementType::DendriteInhibitory);
    const DeviceArray<CudaConfig::synaptic_count_type> to_delete_inhibitory_dendrites{ std::span<const CudaConfig::synaptic_count_type>{ raw_inh } };
    cudaDeviceSynchronize_bridge();
    const auto deleted_inhibitory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Inhibitory, to_delete_inhibitory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    network_graph->rebuild();

    const auto deleted_dendrites = deleted_excitatory_dendrites + deleted_inhibitory_dendrites;

    return { deleted_axons, deleted_dendrites };
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> SynapseDeletionFinderGPU::find_synapses_to_delete(const ElementType element_type, const std::optional<SignalType> single_signal_type, const std::span<const SignalType> signal_types, const DeviceArray<CudaConfig::synaptic_count_type>& number_deletions) {

    Timers::start(TimerRegion::BLOCK2);

    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    auto d_partition = DeviceArray<CudaConfig::synaptic_count_type>(number_deletions.size() + 1);
    const auto sum_to_delete = compute_sum_and_partition(number_deletions.device_ptr(), d_partition.device_ptr(), number_deletions.size());

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto deletion_requests = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(number_ranks, size_hint);

    Timers::stop_and_add(TimerRegion::BLOCK2);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    auto d_request_rank = DeviceArray<CudaConfig::mpi_rank_type>(sum_to_delete);
    auto d_request_neuron_id = DeviceArray<CudaConfig::number_neurons_type>(sum_to_delete);

    find_synapses_to_delete_random_entry(
        DeletionRequestHandle{ number_deletions.device_ptr(), d_partition.device_ptr(), d_request_rank.device_ptr(), d_request_neuron_id.device_ptr() },
        extra_info->get_gpu_handle(), network_graph->get_gpu_handle(),
        element_type, single_signal_type.value_or(SignalType::Excitatory),
        random_key, sum_to_delete);

    Timers::start(TimerRegion::BLOCK1);

    const auto partition = d_partition.get_device_data();
    const auto request_rank = d_request_rank.get_device_data();
    const auto request_neuron_id = d_request_neuron_id.get_device_data();

    const auto number_neurons = number_deletions.size();
    auto current_neuron_id = 0U;
    for (auto i = 0U; i < sum_to_delete; i++) {
        while (partition[current_neuron_id + 1] <= i) {
            current_neuron_id++;
        }
        RelearnException::check(current_neuron_id < number_neurons, "SynapseDeletionFinderGPU::find_synapses_to_delete: Neuron id {} is too large {}", current_neuron_id, number_neurons);
        const auto other_rank = mpiPP::MPIRank(request_rank[i]);
        const auto other_neuron_id = request_neuron_id[i];
        const auto signal_type = single_signal_type.has_value() ? single_signal_type.value() : signal_types[current_neuron_id];
        const auto other_element_type = get_other_element_type(element_type);
        deletion_requests.append(other_rank, SynapseDeletionRequest{ NeuronID{ current_neuron_id }, NeuronID{ other_neuron_id }, other_element_type, signal_type });
    }
    Timers::stop_and_add(TimerRegion::BLOCK1);

    network_graph->device_was_modified();
    network_graph->device_was_modified();

    // Note: this function only computes deletion REQUESTS (find_synapses_to_delete_random_entry
    // reads number_deletions/writes into local d_partition/d_request_rank/d_request_neuron_id
    // buffers only) -- it never touches connected_elements/vacant_elements, so there is nothing to
    // mark device-modified here. That happens in commit_deletions(), via its own get_cuda_handle().

    return deletion_requests;
}

RelearnTypes::number_synapse_type SynapseDeletionFinderGPU::commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions, [[maybe_unused]] const mpiPP::MPIRank my_rank) {

    Timers::start(TimerRegion::BLOCK1);

    auto indices = std::vector<std::tuple<mpiPP::MPIRank, std::size_t, NeuronID>>{};
    indices.reserve(deletions.get_total_number_requests());
    for (const auto& [other_rank, requests] : deletions) {
        for (auto request_index = 0U; request_index < requests.size(); request_index++) {
            const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] = requests[request_index];
            RelearnException::check(other_neuron_id.is_actual_id(), "BarnesHutCUDA::process_requests: Received synapse creation request for invalid neuron id");
            RelearnException::check(other_neuron_id.get_neuron_id() <= size, "BarnesHutCUDA::process_requests: Target neuron id {} is too high {}", other_neuron_id, size);
            RelearnException::check(my_neuron_id.is_actual_id(), "BarnesHutCUDA::process_requests: Received synapse creation request for invalid neuron id");
            RelearnException::check(my_neuron_id.get_neuron_id() <= size, "BarnesHutCUDA::process_requests: Target neuron id {} is too high {}", my_neuron_id, size);

            indices.emplace_back(other_rank, request_index, my_neuron_id);
        }
    }

    std::sort(indices.begin(), indices.end(), [](const auto& t1, const auto& t2) { return std::get<2>(t1) < std::get<2>(t2); });
    const auto partition = utility::create_partition(indices, [](auto& t) { return std::get<2>(t).get_neuron_id(); }, size + 1);
    const DeviceArray<std::size_t> d_partition(partition);

    std::vector<CudaConfig::mpi_rank_type> other_ranks{};
    std::vector<CudaConfig::number_neurons_type> other_neuron_ids{};
    std::vector<ElementType> my_element_types{};
    std::vector<SignalType> my_signal_types{};

    for (const auto& [other_rank, request_index, _other_neuron] : indices) {
        const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] = deletions.get_request(
            other_rank, request_index);
        other_neuron_ids.emplace_back(other_neuron_id.get_neuron_id());
        other_ranks.emplace_back(other_rank.get_rank());
        my_element_types.emplace_back(element_type);
        my_signal_types.emplace_back(signal_type);
    }

    const DeviceArray<CudaConfig::mpi_rank_type> d_other_ranks(other_ranks);
    const DeviceArray<CudaConfig::number_neurons_type> d_other_neuron_ids(other_neuron_ids);
    const DeviceArray<ElementType> d_my_element_types(my_element_types);
    const DeviceArray<SignalType> d_my_signal_types(my_signal_types);

    const auto axon_handle = synaptic_elements->get_axons()->get_cuda_handle();
    const auto den_exc_handle = synaptic_elements->get_dendrites()->get_cuda_handle(SignalType::Excitatory);
    const auto den_inh_handle = synaptic_elements->get_dendrites()->get_cuda_handle(SignalType::Inhibitory);

    Timers::stop_and_add(TimerRegion::BLOCK1);

    commit_deletions_entry(extra_info->get_gpu_handle(),
                           axon_handle, den_exc_handle, den_inh_handle,
                           network_graph->get_gpu_handle(),
                           DeletionCommitHandle{ d_partition.device_ptr(), d_other_ranks.device_ptr(), d_other_neuron_ids.device_ptr(),
                                                 d_my_element_types.device_ptr(), d_my_signal_types.device_ptr() });

    // axon_handle/den_exc_handle/den_inh_handle above already requested non-const device pointers
    // via get_cuda_handle(), which already marked grown/delta/vacant/connected device-modified.
    network_graph->device_was_modified();

    return deletions.get_total_number_requests();
}
