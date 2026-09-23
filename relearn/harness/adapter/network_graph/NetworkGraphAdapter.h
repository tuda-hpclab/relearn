#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NetworkGraph.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "types/BasicTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/random/random_factory.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/action/insert.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/traits.hpp>
#include <range/v3/utility/tuple_algorithm.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <concepts>
#include <map>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace detail {
const auto to_neuron_id = utility::element<0>;
const auto to_edge_weight = utility::element<1>;
const auto to_rank_neuron_id_and_weight_pair = [](const auto my_rank) {
    return [my_rank](const auto local_edge) {
        return std::pair{ RankNeuronId{ my_rank, to_neuron_id(local_edge) },
                          to_edge_weight(local_edge) };
    };
};
} // namespace detail

class NetworkGraphAdapter {
public:
    template <typename T, typename synapse_weight>
    static void erase_empty(std::map<T, synapse_weight>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            if (iterator->second == 0) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    template <typename T1, typename T2, typename synapse_weight>
    static void erase_empties(std::map<T1, std::map<T2, synapse_weight>>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            erase_empty<T2>(iterator->second);

            if (iterator->second.empty()) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    template <typename SynapseType>
    static void add_synapses(NetworkGraph& ng, const std::vector<SynapseType>& synapses) {
        for (const auto& synapse : synapses) {
            ng.add_synapse(synapse);
        }
    }

    static auto transform_synapses(const std::vector<PlasticLocalSynapse>& synapses) {
        std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>> incoming_edges{};
        std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>> outgoing_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;

            incoming_edges[target_id][source_id] += weight;
            outgoing_edges[source_id][target_id] += weight;
        }

        erase_empties(incoming_edges);
        erase_empties(outgoing_edges);

        return std::pair{ incoming_edges, outgoing_edges };
    }

    static auto transform_synapses(const std::vector<StaticLocalSynapse>& synapses) {
        std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>> incoming_edges{};
        std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>> outgoing_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;

            incoming_edges[target_id][source_id] += weight;
            outgoing_edges[source_id][target_id] += weight;
        }

        erase_empties(incoming_edges);
        erase_empties(outgoing_edges);

        return std::pair{ incoming_edges, outgoing_edges };
    }

    static auto transform_synapses(const std::vector<PlasticDistantInSynapse>& synapses) {
        std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> incoming_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;
            incoming_edges[target_id][source_id] += weight;
        }

        erase_empties(incoming_edges);
        return incoming_edges;
    }

    static auto transform_synapses(const std::vector<StaticDistantInSynapse>& synapses) {
        std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>> incoming_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;
            incoming_edges[target_id][source_id] += weight;
        }

        erase_empties(incoming_edges);
        return incoming_edges;
    }

    static auto transform_synapses(const std::vector<PlasticDistantOutSynapse>& synapses) {
        std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>> incoming_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;
            incoming_edges[source_id][target_id] += weight;
        }

        erase_empties(incoming_edges);
        return incoming_edges;
    }

    static auto transform_synapses(const std::vector<StaticDistantOutSynapse>& synapses) {
        std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>> incoming_edges{};

        for (const auto& synapse : synapses) {
            const auto& [target_id, source_id, weight] = synapse;
            incoming_edges[source_id][target_id] += weight;
        }

        erase_empties(incoming_edges);
        return incoming_edges;
    }

    template <ranges::range LocalEdgeRangeType, ranges::range DistantEdgeRangeType>
        requires std::same_as<typename ranges::range_value_t<LocalEdgeRangeType>::second_type, typename ranges::range_value_t<DistantEdgeRangeType>::second_type>
    [[nodiscard]] static std::vector<std::pair<RankNeuronId, typename ranges::range_value_t<LocalEdgeRangeType>::second_type>> get_all_edges(const LocalEdgeRangeType& all_local_edges, const DistantEdgeRangeType& all_distant_edges, const mpiPP::MPIRank my_rank, const SignalType signal_type) {
        switch (signal_type) {
        case SignalType::Excitatory:
            return ranges::views::concat(
                       all_local_edges | ranges::views::filter(utility::greater(0), detail::to_edge_weight) | ranges::views::transform(detail::to_rank_neuron_id_and_weight_pair(my_rank)),
                       all_distant_edges | ranges::views::filter(utility::greater(0), detail::to_edge_weight))
                   | ranges::to_vector;

        case SignalType::Inhibitory:
            return ranges::views::concat(
                       all_local_edges | ranges::views::filter(utility::less(0), detail::to_edge_weight) | ranges::views::transform(detail::to_rank_neuron_id_and_weight_pair(my_rank)),
                       all_distant_edges | ranges::views::filter(utility::less(0), detail::to_edge_weight))
                   | ranges::to_vector;
        }
        RelearnException::fail("NetworkGraphFactory::construct_empty_network_graph: Unknown SignalType {}", signal_type);
    }

    template <ranges::range LocalEdgeRangeType, ranges::range DistantEdgeRangeType>
        requires std::same_as<typename ranges::range_value_t<LocalEdgeRangeType>::second_type, typename ranges::range_value_t<DistantEdgeRangeType>::second_type>
    [[nodiscard]] static std::vector<std::pair<RankNeuronId, typename ranges::range_value_t<LocalEdgeRangeType>::second_type>> get_all_edges(const LocalEdgeRangeType& all_local_edges, const DistantEdgeRangeType& all_distant_edges, const mpiPP::MPIRank my_rank) {
        return ranges::views::concat(
                   all_local_edges | ranges::views::transform(detail::to_rank_neuron_id_and_weight_pair(my_rank)),
                   all_distant_edges)
               | ranges::to_vector;
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>> get_all_plastic_in_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id, const SignalType signal_type) {
        const auto& [all_distant_edges, _1] = ng.get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [all_local_edges, _2] = ng.get_local_in_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank, signal_type);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>> get_all_plastic_out_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id, const SignalType signal_type) {
        const auto& [all_distant_edges, _1] = ng.get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [all_local_edges, _2] = ng.get_local_out_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank, signal_type);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>> get_all_plastic_in_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [all_distant_edges, _1] = ng.get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [all_local_edges, _2] = ng.get_local_in_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>> get_all_plastic_out_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [all_distant_edges, _1] = ng.get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [all_local_edges, _2] = ng.get_local_out_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_static_in_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id, const SignalType signal_type) {
        const auto& [_1, all_distant_edges] = ng.get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [_2, all_local_edges] = ng.get_local_in_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank, signal_type);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_static_out_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id, const SignalType signal_type) {
        const auto& [_1, all_distant_edges] = ng.get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [_2, all_local_edges] = ng.get_local_out_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank, signal_type);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_static_in_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [_1, all_distant_edges] = ng.get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [_2, all_local_edges] = ng.get_local_in_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_static_out_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [_1, all_distant_edges] = ng.get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [_2, all_local_edges] = ng.get_local_out_edges(neuron_id.get_neuron_id());

        return get_all_edges(all_local_edges, all_distant_edges, my_rank);
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_out_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [plastic_distant_edges, static_distant_edges] = ng.get_distant_out_edges(neuron_id.get_neuron_id());
        const auto& [plastic_local_edges, static_local_edges] = ng.get_local_out_edges(neuron_id.get_neuron_id());

        std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> all_edges{};
        all_edges.reserve(plastic_distant_edges.size() + plastic_local_edges.size() + static_distant_edges.size() + static_local_edges.size());

        auto local_to_distant_edges = [my_rank](const std::pair<NeuronID, RelearnTypes::static_synapse_weight>& pair) { return std::make_pair(RankNeuronId{ my_rank, pair.first }, pair.second); };

        // The plastic edges count their connections, so their weight has to be widened to the one of the static edges.
        auto plastic_to_static_edges = [](const std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>& pair) { return std::make_pair(pair.first, static_cast<RelearnTypes::static_synapse_weight>(pair.second)); };
        auto local_plastic_to_distant_static_edges = [my_rank](const std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>& pair) { return std::make_pair(RankNeuronId{ my_rank, pair.first }, static_cast<RelearnTypes::static_synapse_weight>(pair.second)); };

        std::ranges::copy(static_distant_edges, std::back_inserter(all_edges));
        std::ranges::transform(plastic_distant_edges, std::back_inserter(all_edges), plastic_to_static_edges);

        std::ranges::transform(static_local_edges, std::back_inserter(all_edges), local_to_distant_edges);
        std::ranges::transform(plastic_local_edges, std::back_inserter(all_edges), local_plastic_to_distant_static_edges);

        return all_edges;
    }

    [[nodiscard]] static std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> get_all_in_edges(const NetworkGraph& ng, const mpiPP::MPIRank my_rank, const NeuronID neuron_id) {
        const auto& [plastic_distant_edges, static_distant_edges] = ng.get_distant_in_edges(neuron_id.get_neuron_id());
        const auto& [plastic_local_edges, static_local_edges] = ng.get_local_in_edges(neuron_id.get_neuron_id());

        std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> all_edges{};
        all_edges.reserve(plastic_distant_edges.size() + plastic_local_edges.size() + static_distant_edges.size() + static_local_edges.size());

        auto local_to_distant_edges = [my_rank](const std::pair<NeuronID, RelearnTypes::static_synapse_weight>& pair) { return std::make_pair(RankNeuronId{ my_rank, pair.first }, pair.second); };

        // The plastic edges count their connections, so their weight has to be widened to the one of the static edges.
        auto plastic_to_static_edges = [](const std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>& pair) { return std::make_pair(pair.first, static_cast<RelearnTypes::static_synapse_weight>(pair.second)); };
        auto local_plastic_to_distant_static_edges = [my_rank](const std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>& pair) { return std::make_pair(RankNeuronId{ my_rank, pair.first }, static_cast<RelearnTypes::static_synapse_weight>(pair.second)); };

        std::ranges::copy(static_distant_edges, std::back_inserter(all_edges));
        std::ranges::transform(plastic_distant_edges, std::back_inserter(all_edges), plastic_to_static_edges);

        std::ranges::transform(static_local_edges, std::back_inserter(all_edges), local_to_distant_edges);
        std::ranges::transform(plastic_local_edges, std::back_inserter(all_edges), local_plastic_to_distant_static_edges);

        return all_edges;
    }

    /**
     * @brief Adds missing distant ingoing connections to the network_graphs if there is a corresponding distant outgoing connections
     * @param network_graphs Vector of network_graphs. Rank i has network_graphs[i]
     * @param num_neurons Number of neurons per rank
     */
    static bool harmonize_network_graphs_from_different_ranks(std::vector<std::shared_ptr<NetworkGraph>> network_graphs, const RelearnTypes::number_neurons_type num_neurons) {
        for (auto rank = 0; static_cast<std::size_t>(rank) < network_graphs.size(); rank++) {
            const auto cur_network_graph = network_graphs[static_cast<std::size_t>(rank)];

            for (auto source_id = 0ULL; source_id < num_neurons; source_id++) {
                const auto [distant_out_edges_plastic, distant_out_edges_static] = cur_network_graph->get_distant_out_edges(source_id);
                const auto source_rank_id = RankNeuronId{ mpiPP::MPIRank(rank), NeuronID(source_id) };

                for (const auto& [target, weight] : distant_out_edges_plastic) {
                    const auto target_rank = target.get_rank().get_rank();
                    if (rank == target_rank) {
                        return false;
                    }

                    auto& other_network_graph = network_graphs[static_cast<std::size_t>(target_rank)];
                    other_network_graph->add_synapse(PlasticDistantInSynapse(target.get_neuron_id(), source_rank_id, weight));
                }

                for (const auto& [target, weight] : distant_out_edges_static) {
                    const auto target_rank = target.get_rank().get_rank();
                    if (rank == target_rank) {
                        return false;
                    }

                    auto& other_network_graph = network_graphs[static_cast<std::size_t>(target_rank)];
                    other_network_graph->add_synapse(StaticDistantInSynapse(target.get_neuron_id(), source_rank_id, weight));
                }
            }
        }

        return true;
    }

    /**
     * @brief Checks if the signal_types of synapses corresponds with the weights in the network graph and if the distant outgoing and ingoing edges match
     * @param network_graphs Vector of network_graphs. Rank i has network_graphs[i]
     * @param signal_types Vector of vector of signal types. Neuron j on rank i has signal_type[i][j]
     * @param num_neurons Number of neurons per rank
     */
    static bool check_validity_of_network_graphs(std::vector<std::shared_ptr<NetworkGraph>> network_graphs, const std::vector<std::vector<SignalType>>& signal_types, const RelearnTypes::number_neurons_type num_neurons) {
        for (auto rank = 0ULL; rank < network_graphs.size(); rank++) {
            const auto cur_network_graph = network_graphs[rank];

            for (auto neuron_id = 0ULL; neuron_id < num_neurons; neuron_id++) {
                const auto& [local_out_edges_pastic, local_out_edges_static] = cur_network_graph->get_local_out_edges(neuron_id);
                for (const auto& [target, weight] : local_out_edges_pastic) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][neuron_id]) {
                        return false;
                    }
                }
                for (const auto& [target, weight] : local_out_edges_static) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][neuron_id]) {
                        return false;
                    }
                }

                const auto [distant_out_edges_plastic, distant_out_edges_static] = cur_network_graph->get_distant_out_edges(neuron_id);
                for (const auto& [target, weight] : distant_out_edges_plastic) {
                    const auto target_rank = target.get_rank().get_rank_cast();
                    if (rank == target_rank) {
                        return false;
                    }

                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][neuron_id]) {
                        return false;
                    }

                    const auto& other_network_graph = network_graphs[target_rank];
                    const auto& [other_distant_in_edges, _] = other_network_graph->get_distant_in_edges(target.get_neuron_id().get_neuron_id());
                    bool found_edge = false;
                    for (const auto& [other_source, other_weight] : other_distant_in_edges) {
                        if (other_source.get_rank().get_rank_cast() == rank && other_source.get_neuron_id().get_neuron_id() == neuron_id) {
                            found_edge = true;
                            break;
                        }
                    }

                    if (!found_edge) {
                        return false;
                    }
                }
                for (const auto& [target, weight] : distant_out_edges_static) {
                    const auto target_rank = target.get_rank().get_rank_cast();
                    if (rank == target_rank) {
                        return false;
                    }

                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][neuron_id]) {
                        return false;
                    }

                    const auto& other_network_graph = network_graphs[target_rank];
                    const auto& [_, other_distant_in_edges] = other_network_graph->get_distant_in_edges(target.get_neuron_id().get_neuron_id());
                    bool found_edge = false;
                    for (const auto& [other_source, other_weight] : other_distant_in_edges) {
                        if (other_source.get_rank().get_rank_cast() == rank && other_source.get_neuron_id().get_neuron_id() == neuron_id) {
                            found_edge = true;
                            break;
                        }
                    }

                    if (!found_edge) {
                        return false;
                    }
                }

                const auto& [local_in_edges_plastic, local_in_edges_static] = cur_network_graph->get_local_in_edges(neuron_id);
                for (const auto& [source, weight] : local_in_edges_plastic) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][source.get_neuron_id()]) {
                        return false;
                    }
                }
                for (const auto& [source, weight] : local_in_edges_static) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[rank][source.get_neuron_id()]) {
                        return false;
                    }
                }

                const auto [distant_in_edges_plastic, distant_in_edges_static] = cur_network_graph->get_distant_in_edges(neuron_id);
                for (const auto& [source, weight] : distant_in_edges_plastic) {
                    const auto source_rank = source.get_rank().get_rank_cast();
                    if (rank == source_rank) {
                        return false;
                    }

                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[source_rank][source.get_neuron_id().get_neuron_id()]) {
                        return false;
                    }

                    const auto& other_network_graph = network_graphs[source_rank];
                    const auto& [other_distant_out_edges, _] = other_network_graph->get_distant_out_edges(source.get_neuron_id().get_neuron_id());
                    bool found_edge = false;
                    for (const auto& [other_target, other_weight] : other_distant_out_edges) {
                        if (other_target.get_rank().get_rank_cast() == rank && other_target.get_neuron_id().get_neuron_id() == neuron_id) {
                            if (weight != other_weight) {
                                return false;
                            }
                            found_edge = true;
                            break;
                        }
                    }

                    if (!found_edge) {
                        return false;
                    }
                }
                for (const auto& [source, weight] : distant_in_edges_static) {
                    const auto source_rank = source.get_rank().get_rank_cast();
                    if (rank == source_rank) {
                        return false;
                    }

                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    if (signal_type != signal_types[source_rank][source.get_neuron_id().get_neuron_id()]) {
                        return false;
                    }

                    const auto& other_network_graph = network_graphs[source_rank];
                    const auto& [_, other_distant_out_edges] = other_network_graph->get_distant_out_edges(source.get_neuron_id().get_neuron_id());
                    bool found_edge = false;
                    for (const auto& [other_target, other_weight] : other_distant_out_edges) {
                        if (other_target.get_rank().get_rank_cast() == rank && other_target.get_neuron_id().get_neuron_id() == neuron_id) {
                            if (weight != other_weight) {
                                return false;
                            }
                            found_edge = true;
                            break;
                        }
                    }

                    if (!found_edge) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    static RelearnTypes::plastic_synapse_weight get_weight(const SignalType signal_type) {
        return signal_type == SignalType::Excitatory ? 1 : -1;
    }

    static void connect_all_to_all(std::shared_ptr<NetworkGraph> network_graph, const std::span<const SignalType> signal_types) {
        const auto number_neurons = network_graph->get_number_neurons();

        const auto not_the_same_id = [](const auto& id_pair) {
            return utility::element<0>(id_pair) != utility::element<1>(id_pair);
        };

        const auto func = [&network_graph, signal_types](const auto& id_pair) {
            const auto& [source_id, target_id] = id_pair;
            const auto signal_type = signal_types[source_id.get_neuron_id()];
            const auto weight = get_weight(signal_type);
            const auto synapse = PlasticLocalSynapse{ target_id, source_id, weight };

            network_graph->add_synapse(synapse);
        };

        ranges::for_each(
            ranges::views::cartesian_product(NeuronIDRange::range(number_neurons), NeuronIDRange::range(number_neurons))
                | ranges::views::filter(not_the_same_id),
            func);
    }

    static void connect_to_n_th_other(const std::shared_ptr<NetworkGraph>& network_graph, const std::span<const SignalType> signal_types, const std::size_t offset) {
        const auto number_neurons = network_graph->get_number_neurons();

        for (auto neuron_id = NeuronID::value_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
            const auto target_id = (neuron_id + offset) % number_neurons;

            const auto signal_type = signal_types[neuron_id];
            const auto weight = get_weight(signal_type);
            const auto synapse = PlasticLocalSynapse{ NeuronID(target_id), NeuronID(neuron_id), weight };
            network_graph->add_synapse(synapse);
        }
    }
};
