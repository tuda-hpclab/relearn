/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinderCPU.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/SynapseTypes.h"
#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/filter.hpp>

std::pair<RelearnTypes::number_synapse_type, RelearnTypes::number_synapse_type> SynapseDeletionFinderCPU::delete_synapses() {
    auto deletion_helper_axons = [this](const ElementType element_type, const std::span<const SignalType> signal_types,
                                        const std::span<const counter_type> to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, signal_types, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(
            outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    auto deletion_helper_dendrites = [this](const ElementType element_type, SignalType signal_type,
                                            const std::span<const RelearnTypes::counter_type> to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, signal_type, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(
            outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_axons = synaptic_elements->commit_updates(SynapticElementType::Axon);
    const auto deleted_axons = deletion_helper_axons(ElementType::Axon, synaptic_elements->get_signal_types(),
                                                     to_delete_axons);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_excitatory_dendrites = synaptic_elements->commit_updates(
        SynapticElementType::DendriteExcitatory);
    const auto deleted_excitatory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Excitatory,
                                                                        to_delete_excitatory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_inhibitory_dendrites = synaptic_elements->commit_updates(
        SynapticElementType::DendriteInhibitory);

    const auto deleted_inhibitory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Inhibitory,
                                                                        to_delete_inhibitory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    const auto deleted_dendrites = deleted_excitatory_dendrites + deleted_inhibitory_dendrites;

    return { deleted_axons, deleted_dendrites };
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest>
SynapseDeletionFinderCPU::find_synapses_to_delete(const ElementType element_type,
                                                  const std::span<const SignalType> signal_types,
                                                  std::span<const counter_type> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto deletion_requests = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (const auto& neuron_id : NeuronIDRange::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto signal_type = signal_types[local_neuron_id];
        const auto affected_neuron_ids = find_synapses_on_neuron(neuron_id, element_type, signal_type,
                                                                 num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            const auto psd = SynapseDeletionRequest(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(
                    PlasticDistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(
                    PlasticDistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> SynapseDeletionFinderCPU::find_synapses_to_delete(const ElementType element_type, SignalType signal_type, std::span<const counter_type> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto deletion_requests = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (const auto& neuron_id : NeuronIDRange::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto affected_neuron_ids = find_synapses_on_neuron(neuron_id, element_type, signal_type,
                                                                 num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            const auto psd = SynapseDeletionRequest(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(
                    PlasticDistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(
                    PlasticDistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

RelearnTypes::number_synapse_type
SynapseDeletionFinderCPU::commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions,
                                           const mpiPP::MPIRank my_rank) {
    auto num_synapses_deleted = RelearnTypes::number_synapse_type{ 0 };

    for (const auto& [other_rank, requests] : deletions) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(PlasticLocalSynapse(other_neuron_id, my_neuron_id, weight));
                    extra_info->mark_deletion(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight);
                } else {
                    network_graph->add_synapse(PlasticLocalSynapse(my_neuron_id, other_neuron_id, weight));
                    extra_info->mark_deletion(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight);
                }
            } else {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(
                        PlasticDistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, weight));
                } else {
                    network_graph->add_synapse(
                        PlasticDistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), weight));

                    extra_info->mark_deletion(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight);
                }
            }

            const auto synaptic_element_type = get_synaptic_element_type(get_other_element_type(element_type),
                                                                         signal_type);
            synaptic_elements->disconnect_elements(1, my_neuron_id.get_neuron_id(), synaptic_element_type);
        }
    }

    return num_synapses_deleted;
}
