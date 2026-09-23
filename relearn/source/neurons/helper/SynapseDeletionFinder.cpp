/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinder.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/ProbabilityPicker.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <range/v3/action/insert.hpp>
#include <range/v3/algorithm/lower_bound.hpp>
#include <range/v3/numeric/accumulate.hpp>
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

std::vector<RankNeuronId> RandomSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const counter_type num_synapses_to_delete) {
    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    const auto& current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses,
                            "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& drawn_indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::SynapseDeletion,
                                                                         num_synapses_to_delete, number_synapses);

    return drawn_indices | ranges::views::transform(utility::lookup(current_synapses)) | ranges::to_vector;
}

#ifndef RELEARN_CUDA_ENABLED
RelearnTypes::comm_map_deletion<SynapseDeletionRequest> InverseLengthSynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, const std::span<const SignalType> signal_types,
                                                                                                                    std::span<const counter_type> number_deletions) {
    auto found_partners = find_partners_to_locate(element_type, signal_types, number_deletions);
    auto requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(found_partners);

    auto my_positions = extra_info->get_positions_for(requests);
    auto responses = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(my_positions);

    this->partners = std::move(found_partners);
    this->positions = std::move(responses);

    return SynapseDeletionFinder::find_synapses_to_delete(element_type, signal_types, number_deletions);
}

RelearnTypes::comm_map_deletion<SynapseDeletionRequest> InverseLengthSynapseDeletionFinder::find_synapses_to_delete(const ElementType element_type, SignalType signal_type,
                                                                                                                    std::span<const counter_type> number_deletions) {
    auto found_partners = find_partners_to_locate(element_type, signal_type, number_deletions);
    auto requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(found_partners);

    auto my_positions = extra_info->get_positions_for(requests);
    auto responses = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(my_positions);

    this->partners = std::move(found_partners);
    this->positions = std::move(responses);

    return SynapseDeletionFinder::find_synapses_to_delete(element_type, signal_type, number_deletions);
}
#endif

RelearnTypes::comm_map_deletion<NeuronID> InverseLengthSynapseDeletionFinder::find_partners_to_locate(const ElementType element_type, const std::span<const SignalType> signal_types,
                                                                                                      std::span<const counter_type> number_deletions) {

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

    for (const auto neuron_id : NeuronIDRange::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
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
            const auto& neuron_partners = network_graph->get_all_plastic_partners_incoming(local_neuron_id,
                                                                                           signal_type);
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
                                                                                                      std::span<const counter_type> number_deletions) {

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

    for (const auto neuron_id : NeuronIDRange::range(number_neurons) | ranges::views::filter([this](const auto& neuron_id) { return extra_info->does_update_plasticity(neuron_id); })) {
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
            const auto& neuron_partners = network_graph->get_all_plastic_partners_incoming(local_neuron_id,
                                                                                           signal_type);
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

std::vector<RankNeuronId> InverseLengthSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const counter_type num_synapses_to_delete) {
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses,
                            "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& my_position = extra_info->get_position(neuron_id);

    const auto get_probabilities = [&my_position, &neuron_id, this](const std::vector<RankNeuronId>& others) -> std::vector<RelearnTypes::attraction_type> {
        return others | ranges::views::transform([&my_position, &neuron_id, this](const RankNeuronId& rni) {
                   const auto& [other_rank, other_id] = rni;
                   if (neuron_id == other_id) {
                       // In case a neuron has a synapse to itself, return 1.0
                       return RelearnTypes::attraction_type{ 1 };
                   }

                   const auto& relevant_ids = partners.get_requests(other_rank);

                   const auto pos = ranges::lower_bound(relevant_ids, other_id);
                   RelearnException::check(pos != relevant_ids.end(),
                                           "InverseLengthSynapseDeletionFinder::find_synapses_on_neuron: Did not find the id {} in the communication (map) at rank {}",
                                           other_id, other_rank);

                   const auto distance = std::distance(relevant_ids.begin(), pos);

                   const auto other_pos = positions.get_request(other_rank, static_cast<std::size_t>(distance));

                   const auto& diff = other_pos - my_position;
                   const auto euclidean_distance = diff.calculate_2_norm<RelearnTypes::attraction_type>();

                   return RelearnTypes::attraction_type{ 1 } / euclidean_distance;
               })
               | ranges::to_vector;
    };

    auto affected_neurons = std::vector<RankNeuronId>{};
    affected_neurons.reserve(num_synapses_to_delete);

    auto probabilities = get_probabilities(current_synapses);
    for (auto i = 0U; i < num_synapses_to_delete; i++) {
        const auto idx = ProbabilityPicker::pick_target(probabilities, RandomHolderKey::SynapseDeletion);

        affected_neurons.emplace_back(current_synapses[idx]);
        probabilities[idx] = RelearnTypes::attraction_type{ 0 };
    }

    return affected_neurons;
}

// std::vector<RankNeuronId> CoActivationSynapseDeletionFinder::find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, counter_type num_synapses_to_delete) {
//     // Only do something if necessary
//     if (0 == num_synapses_to_delete) {
//         return {};
//     }
//
//     auto current_synapses = register_synapses(neuron_id, element_type, signal_type);
//
//     const auto number_synapses = current_synapses.size();
//
//     RelearnException::check(num_synapses_to_delete <= number_synapses,
//                             "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > distant_synapses.size()");
//
//     RandomHolder::shuffle(RandomHolderKey::SynapseDeletion, current_synapses.begin(), current_synapses.end());
//
//     auto co_activations = std::vector<std::pair<RankNeuronId, double>>{};
//     co_activations.reserve(number_synapses);
//     for (const auto &rank_neuron_id: current_synapses) {
//         double co_activation = std::numeric_limits<double>::quiet_NaN();
//         if (element_type == ElementType::Axon) {
//             co_activation = calculate_co_activation(fired_status_recorder->get_fire_history(neuron_id),
//                                                     fired_status_recorder->get_fire_history(rank_neuron_id));
//         } else {
//             co_activation = calculate_co_activation(fired_status_recorder->get_fire_history(rank_neuron_id),
//                                                     fired_status_recorder->get_fire_history(neuron_id));
//         }
//         co_activations.emplace_back(rank_neuron_id, co_activation);
//     }
//
//     std::ranges::sort(co_activations, [](const auto &p1, const auto &p2) { return p1.second < p2.second; });
//
//     auto affected_neurons = std::vector<RankNeuronId>{};
//     affected_neurons.reserve(num_synapses_to_delete);
//
//     for (auto i = 0U; i < num_synapses_to_delete; i++) {
//         affected_neurons.push_back(co_activations[i].first);
//     }
//
//     return affected_neurons;
// }
//
// double CoActivationSynapseDeletionFinder::calculate_co_activation(const boost::dynamic_bitset<> &pre_synaptic,
//                                                                   const boost::dynamic_bitset<> &post_synaptic) {
//     RelearnException::check(pre_synaptic.size() == post_synaptic.size(),
//                             "SynapseDeletionFinder::calculate_co_activation: Fire histories have different sizes");
//
//     auto intersection = 0U;
//     for (auto i = 0U; i < pre_synaptic.size(); i++) {
//         if (static_cast<bool>(FiredStatus::Fired) == pre_synaptic[i] && pre_synaptic[i] == post_synaptic[i]) {
//             intersection++;
//         }
//     }
//     return static_cast<double>(intersection) / static_cast<double>(pre_synaptic.size());
// }
//
RandomSynapseDeletionFinder::RandomSynapseDeletionFinder() = default;

void RandomSynapseDeletionFinder::init(const RelearnTypes::number_neurons_type number_neurons) {
    SynapseDeletionFinder::init(number_neurons);
}
