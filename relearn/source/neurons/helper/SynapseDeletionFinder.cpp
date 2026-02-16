/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinder.h"

#include "Types.h"
#include "Types3.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/NeuronID.h"
#include "util/ProbabilityPicker.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIAdvancedCommunicationPatterns.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <range/v3/action/insert.hpp>
#include <range/v3/algorithm/lower_bound.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

void SynapseDeletionFinder::set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
    const auto full = new_extra_info != nullptr;
    RelearnException::check(full, "SynapseDeletionFinder::set_extra_infos: new_extra_info is empty");

    extra_info = std::move(new_extra_info);
}

void SynapseDeletionFinder::set_fired_status_recorder(std::shared_ptr<FiredStatusRecorder> new_fired_status_recorder) {
    const auto full = new_fired_status_recorder != nullptr;
    RelearnException::check(full, "SynapseDeletionFinder::set_fired_status_recorder: new_fired_status_recorder is empty");

    fired_status_recorder = std::move(new_fired_status_recorder);
}

std::pair<std::uint64_t, std::uint64_t> SynapseDeletionFinder::delete_synapses() {
    auto deletion_helper_axons = [this](const ElementType element_type, const std::span<const SignalType> signal_types, const std::span<const unsigned int> to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, signal_types, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        // const auto incoming_deletion_requests = MPIWrapper::exchange_requests(outgoing_deletion_requests, first_map, second_map);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    auto deletion_helper_dendrites = [this](const ElementType element_type, SignalType signal_type, const std::span<const unsigned int> to_delete) {
        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(element_type, signal_type, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        // const auto incoming_deletion_requests = MPIWrapper::exchange_requests(outgoing_deletion_requests, first_map, second_map);
        const auto incoming_deletion_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_elements = commit_deletions(incoming_deletion_requests, mpiPP::MPIInfo::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        return newly_freed_elements;
    };

    Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_axons = synaptic_elements->commit_updates(SynapticElementType::Axon);
    const auto deleted_axons = deletion_helper_axons(ElementType::Axon, synaptic_elements->get_signal_types(), to_delete_axons);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_excitatory_dendrites = synaptic_elements->commit_updates(SynapticElementType::DendriteExcitatory);
    const auto deleted_excitatory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Excitatory, to_delete_excitatory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    const auto to_delete_inhibitory_dendrites = synaptic_elements->commit_updates(SynapticElementType::DendriteInhibitory);
    const auto deleted_inhibitory_dendrites = deletion_helper_dendrites(ElementType::Dendrite, SignalType::Inhibitory, to_delete_inhibitory_dendrites);
    Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

    Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    const auto deleted_dendrites = deleted_excitatory_dendrites + deleted_inhibitory_dendrites;

    return { deleted_axons, deleted_dendrites };
}

std::vector<mpiPP::MPIRank> SynapseDeletionFinder::build_rank_whitelist() const {
    auto whitelist = std::vector<mpiPP::MPIRank>{};

    const auto& distant_count = network_graph->plastic_network_graph.get_distant_count();
    whitelist.reserve(distant_count.size());
    for (auto i = 0U; i < distant_count.size(); i++) {
        if (distant_count[i] > 0) {
            whitelist.emplace_back(i);
        }
    }

    return whitelist;
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> SynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, const std::span<const SignalType> signal_types, std::span<const unsigned int> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto deletion_requests = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (const auto& neuron_id : NeuronID::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
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
        const auto affected_neuron_ids = find_synapses_on_neuron(neuron_id, element_type, signal_type, num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            const auto psd = SynapseDeletionRequest(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(PlasticDistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(PlasticDistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> SynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, SignalType signal_type, std::span<const unsigned int> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto deletion_requests = RelearnTypes::comm_map_deletion<SynapseDeletionRequest>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (const auto& neuron_id : NeuronID::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto affected_neuron_ids = find_synapses_on_neuron(neuron_id, element_type, signal_type, num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            const auto psd = SynapseDeletionRequest(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(PlasticDistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(PlasticDistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

std::uint64_t SynapseDeletionFinder::commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions, const mpiPP::MPIRank my_rank) {
    auto num_synapses_deleted = std::uint64_t{ 0 };

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

            const auto synaptic_element_type = get_synaptic_element_type(get_other_element_type(element_type), signal_type);
            synaptic_elements->disconnect_elements(1, my_neuron_id.get_neuron_id(), synaptic_element_type);
        }
    }

    return num_synapses_deleted;
}

std::vector<RankNeuronId> SynapseDeletionFinder::register_synapses(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type) {
    auto register_out_edges = [](const auto& distant_out_edges, const auto& local_out_edges) {
        auto neuron_ids = std::vector<RankNeuronId>{};
        neuron_ids.reserve((distant_out_edges.size() + local_out_edges.size()) * 2);

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        for (const auto& [rni, weight] : distant_out_edges) {
            /**
             * Create "edge weight" number of synapses and add them to the synapse list
             * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
             */

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [local_neuron_id, weight] : local_out_edges) {
            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, local_neuron_id);
            }
        }

        return neuron_ids;
    };

    auto register_in_edges = [signal_type](const auto& distant_in_edges, const auto& local_in_edges) {
        auto neuron_ids = std::vector<RankNeuronId>{};
        neuron_ids.reserve((distant_in_edges.size() + local_in_edges.size()) * 2);

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();

        for (const auto& [rni, weight] : distant_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [local_neuron_id, weight] : local_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                                    "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, local_neuron_id);
            }
        }

        return neuron_ids;
    };

    auto current_synapses = std::vector<RankNeuronId>{};
    if (element_type == ElementType::Axon) {
        const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id.get_neuron_id());

        current_synapses = register_out_edges(distant_out_edges, local_out_edges);
    } else {
        const auto& [distant_in_edges, _1] = network_graph->get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [local_in_edges, _2] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());

        current_synapses = register_in_edges(distant_in_edges, local_in_edges);
    }

    return current_synapses;
}

std::vector<RankNeuronId> RandomSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const unsigned int num_synapses_to_delete) {
    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    const auto& current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses, "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& drawn_indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::SynapseDeletionFinder, num_synapses_to_delete, number_synapses);

    return drawn_indices | ranges::views::transform(utility::lookup(current_synapses)) | ranges::to_vector;
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> InverseLengthSynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, const std::span<const SignalType> signal_types,
                                                                                                            std::span<const unsigned int> number_deletions) {
    // const auto& [in_ranks_plastic, _1] = network_graph->get_ranks_in_connected();
    // const auto& [out_ranks_plastic, _2] = network_graph->get_ranks_out_connected();

    // const auto& first_map = _synaptic_elements->get_element_type() == ElementType::Axon ? in_ranks_plastic : out_ranks_plastic;
    // const auto& second_map = _synaptic_elements->get_element_type() == ElementType::Axon ? out_ranks_plastic : in_ranks_plastic;

    auto found_partners = find_partners_to_locate(element_type, signal_types, number_deletions);
    // auto requests = MPIWrapper::exchange_requests(partners, first_map, second_map);
    auto requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(found_partners);

    auto my_positions = extra_info->get_positions_for(requests);
    // auto responses = MPIWrapper::exchange_requests(my_positions, second_map, first_map);
    auto responses = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(my_positions);

    this->partners = std::move(found_partners);
    this->positions = std::move(responses);

    return SynapseDeletionFinder::find_synapses_to_delete(element_type, signal_types, number_deletions);
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> InverseLengthSynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, SignalType signal_type,
                                                                                                            std::span<const unsigned int> number_deletions) {
    // const auto& [in_ranks_plastic, _1] = network_graph->get_ranks_in_connected();
    // const auto& [out_ranks_plastic, _2] = network_graph->get_ranks_out_connected();

    // const auto& first_map = _synaptic_elements->get_element_type() == ElementType::Axon ? in_ranks_plastic : out_ranks_plastic;
    // const auto& second_map = _synaptic_elements->get_element_type() == ElementType::Axon ? out_ranks_plastic : in_ranks_plastic;

    auto found_partners = find_partners_to_locate(element_type, signal_type, number_deletions);
    // auto requests = MPIWrapper::exchange_requests(partners, first_map, second_map);
    auto requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(found_partners);

    auto my_positions = extra_info->get_positions_for(requests);
    // auto responses = MPIWrapper::exchange_requests(my_positions, second_map, first_map);
    auto responses = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(my_positions);

    this->partners = std::move(found_partners);
    this->positions = std::move(responses);

    return SynapseDeletionFinder::find_synapses_to_delete(element_type, signal_type, number_deletions);
}

RelearnTypes::comm_map_deletion<NeuronID> InverseLengthSynapseDeletionFinder::find_partners_to_locate(const ElementType element_type, const std::span<const SignalType> signal_types,
                                                                                              std::span<const unsigned int> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto partners_to_locate = RelearnTypes::comm_map_deletion<NeuronID>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return partners_to_locate;
    }

    const auto number_neurons = extra_info->get_size();

    auto all_partners = std::unordered_set<RankNeuronId>{};
    all_partners.reserve(sum_to_delete);

    for (const auto neuron_id : NeuronID::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
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

        if (element_type == ElementType::Axon) {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_outgoing(local_neuron_id);
            ranges::insert(all_partners, neuron_partners);
        } else {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_incoming(local_neuron_id, signal_type);
            ranges::insert(all_partners, neuron_partners);
        }
    }

    for (const auto& [rank, id] : all_partners) {
        partners_to_locate.emplace_back(rank, id);
    }

    for (auto& [rank, requests] : partners_to_locate) {
        std::ranges::sort(requests);
    }

    return partners_to_locate;
}

RelearnTypes::comm_map_deletion<NeuronID> InverseLengthSynapseDeletionFinder::find_partners_to_locate(const ElementType element_type, SignalType signal_type,
                                                                                              std::span<const unsigned int> number_deletions) {

    const auto sum_to_delete = ranges::accumulate(number_deletions, 0U);
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<std::size_t>(number_ranks), number_deletions.size());
    auto partners_to_locate = RelearnTypes::comm_map_deletion<NeuronID>(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return partners_to_locate;
    }

    const auto number_neurons = extra_info->get_size();

    auto all_partners = std::unordered_set<RankNeuronId>{};
    all_partners.reserve(sum_to_delete);

    for (const auto neuron_id : NeuronID::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        if (element_type == ElementType::Axon) {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_outgoing(local_neuron_id);
            ranges::insert(all_partners, neuron_partners);
        } else {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_incoming(local_neuron_id, signal_type);
            ranges::insert(all_partners, neuron_partners);
        }
    }

    for (const auto& [rank, id] : all_partners) {
        partners_to_locate.emplace_back(rank, id);
    }

    for (auto& [rank, requests] : partners_to_locate) {
        std::ranges::sort(requests);
    }

    return partners_to_locate;
}

std::vector<RankNeuronId> InverseLengthSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const unsigned int num_synapses_to_delete) {
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses,
                            "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& my_position = extra_info->get_position(neuron_id);

    const auto get_probabilities = [&my_position, &neuron_id, this](const std::vector<RankNeuronId>& others) -> std::vector<double> {
        return others | ranges::views::transform([&my_position, &neuron_id, this](const RankNeuronId& rni) {
                   const auto& [other_rank, other_id] = rni;
                   if (neuron_id == other_id) {
                       // In case a neuron has a synapse to itself, return 1.0
                       return 1.0;
                   }

                   const auto& relevant_ids = partners.get_requests(other_rank);

                   const auto pos = ranges::lower_bound(relevant_ids, other_id);
                   RelearnException::check(pos != relevant_ids.end(), "InverseLengthSynapseDeletionFinder::find_synapses_on_neuron: Did not find the id {} in the communication (map) at rank {}", other_id, other_rank);

                   const auto distance = std::distance(relevant_ids.begin(), pos);

                   const auto other_pos = positions.get_request(other_rank, static_cast<std::size_t>(distance));

                   const auto& diff = other_pos - my_position;
                   const auto euclidean_distance = diff.calculate_2_norm();

                   return 1.0 / euclidean_distance;
               })
               | ranges::to_vector;
    };

    auto affected_neurons = std::vector<RankNeuronId>{};
    affected_neurons.reserve(num_synapses_to_delete);

    auto probabilities = get_probabilities(current_synapses);
    for (auto i = 0U; i < num_synapses_to_delete; i++) {
        const auto idx = ProbabilityPicker::pick_target(probabilities, RandomHolderKey::SynapseDeletionFinder);

        affected_neurons.emplace_back(current_synapses[idx]);
        probabilities[idx] = 0.0;
    }

    return affected_neurons;
}

std::vector<RankNeuronId> CoActivationSynapseDeletionFinder::find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) {
    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto current_synapses = register_synapses(neuron_id, element_type, signal_type);

    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses,
                            "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > distant_synapses.size()");

    RandomHolder::shuffle(RandomHolderKey::SynapseDeletionFinder, current_synapses.begin(), current_synapses.end());

    auto co_activations = std::vector<std::pair<RankNeuronId, double>>{};
    co_activations.reserve(number_synapses);
    for (const auto& rank_neuron_id : current_synapses) {
        double co_activation = std::numeric_limits<double>::quiet_NaN();
        if (element_type == ElementType::Axon) {
            co_activation = calculate_co_activation(fired_status_recorder->get_fire_history(neuron_id),
                                                    fired_status_recorder->get_fire_history(rank_neuron_id));
        } else {
            co_activation = calculate_co_activation(fired_status_recorder->get_fire_history(rank_neuron_id),
                                                    fired_status_recorder->get_fire_history(neuron_id));
        }
        co_activations.emplace_back(rank_neuron_id, co_activation);
    }

    std::ranges::sort(co_activations, [](const auto& p1, const auto& p2) { return p1.second < p2.second; });

    auto affected_neurons = std::vector<RankNeuronId>{};
    affected_neurons.reserve(num_synapses_to_delete);

    for (auto i = 0U; i < num_synapses_to_delete; i++) {
        affected_neurons.push_back(co_activations[i].first);
    }

    return affected_neurons;
}

double CoActivationSynapseDeletionFinder::calculate_co_activation(const boost::dynamic_bitset<>& pre_synaptic, const boost::dynamic_bitset<>& post_synaptic) {
    RelearnException::check(pre_synaptic.size() == post_synaptic.size(), "SynapseDeletionFinder::calculate_co_activation: Fire histories have different sizes");

    auto intersection = 0U;
    for (auto i = 0U; i < pre_synaptic.size(); i++) {
        if (static_cast<bool>(FiredStatus::Fired) == pre_synaptic[i] && pre_synaptic[i] == post_synaptic[i]) {
            intersection++;
        }
    }
    return static_cast<double>(intersection) / static_cast<double>(pre_synaptic.size());
}
