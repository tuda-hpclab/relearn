/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutInverted.h"

#include "Types.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/Connector.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <utility>

RelearnTypes::comm_map_creation<SynapseCreationRequest> BarnesHutInverted::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    auto* const root = get_octree_root();

    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, number_neurons, disable_flags, vacant_excitatory_dendrites, vacant_inhibitory_dendrites, synapse_creation_requests_outgoing)
    for (auto neuron_id = 0UL; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto id = NeuronID{ neuron_id };

        const auto number_vacant_excitatory_dendrites = vacant_excitatory_dendrites[neuron_id];
        const auto number_vacant_inhibitory_dendrites = vacant_inhibitory_dendrites[neuron_id];

        if (number_vacant_excitatory_dendrites + number_vacant_inhibitory_dendrites == 0) {
            continue;
        }

        const auto& dendrite_position = extra_infos->get_position(id);

        const auto& excitatory_requests = BarnesHutBase<BarnesHutInvertedCell>::find_target_neurons(probability_kernel, node_cache, { my_rank, id }, dendrite_position, number_vacant_excitatory_dendrites, root, ElementType::Axon, SignalType::Excitatory, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : excitatory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        const auto& inhibitory_requests = BarnesHutBase<BarnesHutInvertedCell>::find_target_neurons(probability_kernel, node_cache, { my_rank, id }, dendrite_position, number_vacant_inhibitory_dendrites, root, ElementType::Axon, SignalType::Inhibitory, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : inhibitory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum> BarnesHutInverted::find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& neuron_ids) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto size_hint = std::min(neuron_ids.size(), static_cast<std::size_t>(number_ranks));
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();
    auto* const root = get_octree_root();

    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto& probability_kernel = *this->kernel;

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(probability_kernel, node_cache, root, my_rank, neuron_ids, disable_flags, vacant_excitatory_dendrites, vacant_inhibitory_dendrites, synapse_creation_requests_outgoing)
    for (auto index = 0UL; index < neuron_ids.size(); ++index) {
        const auto id = neuron_ids[index];
        const auto neuron_id = id.get_neuron_id();
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const auto number_vacant_excitatory_dendrites = vacant_excitatory_dendrites[neuron_id];
        const auto number_vacant_inhibitory_dendrites = vacant_inhibitory_dendrites[neuron_id];

        if (number_vacant_excitatory_dendrites + number_vacant_inhibitory_dendrites == 0) {
            continue;
        }

        const auto& dendrite_position = extra_infos->get_position(id);

        const auto& excitatory_requests = BarnesHutBase<BarnesHutInvertedCell>::find_target_neurons(probability_kernel, node_cache, { my_rank, id }, dendrite_position, number_vacant_excitatory_dendrites, root, ElementType::Axon, SignalType::Excitatory, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : excitatory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        const auto& inhibitory_requests = BarnesHutBase<BarnesHutInvertedCell>::find_target_neurons(probability_kernel, node_cache, { my_rank, id }, dendrite_position, number_vacant_inhibitory_dendrites, root, ElementType::Axon, SignalType::Inhibitory, acceptance_criterion);
        for (const auto& [target_rank, creation_request] : inhibitory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return { synapse_creation_requests_outgoing, RequestTypeEnum::SynapseCreationRequest, DirectionEnum::Backward };
}

std::pair<RelearnTypes::comm_map_creation<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantOutSynapses>>
BarnesHutInverted::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return BackwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantInSynapses BarnesHutInverted::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                              const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return BackwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}
