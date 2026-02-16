/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "network_graph_factory.h"

#include "Types.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/SynapticElementType.h"
#include "util/NeuronID.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIRank.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/traits.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/filter.hpp>

#include <memory>
#include <random>
#include <span>
#include <unordered_set>

std::shared_ptr<NetworkGraph> NetworkGraphFactory::construct_empty_network_graph(const NeuronID::value_type number_neurons, const mpiPP::MPIRank mpi_rank, const int number_ranks) {
    auto network_graph = std::make_shared<NetworkGraph>(mpi_rank, number_ranks);
    network_graph->init(number_neurons);
    return network_graph;
}

std::shared_ptr<NetworkGraph> NetworkGraphFactory::construct_all_to_all_network_graph(const NeuronID::value_type number_neurons, const mpiPP::MPIRank mpi_rank) {
    auto ptr = std::make_shared<NetworkGraph>(mpi_rank);
    ptr->init(number_neurons);

    const auto not_the_same_id = [](const auto& id_pair) {
        return utility::element<0>(id_pair) != utility::element<1>(id_pair);
    };

    const auto func = [&ptr](const auto& id_pair) {
        const auto& [source_id, target_id] = id_pair;
        const auto weight = 1;
        const auto ls = PlasticLocalSynapse{ target_id, source_id, weight };

        ptr->add_synapse(ls);
    };

    ranges::for_each(
        ranges::views::cartesian_product(NeuronID::range(number_neurons), NeuronID::range(number_neurons))
            | ranges::views::filter(not_the_same_id),
        func);

    return ptr;
}

std::shared_ptr<NetworkGraph> NetworkGraphFactory::construct_all_to_all_network_graph(const NeuronID::value_type number_neurons, const mpiPP::MPIRank mpi_rank, std::mt19937& mt) {
    auto ptr = std::make_shared<NetworkGraph>(mpi_rank);
    ptr->init(number_neurons);

    const auto not_the_same_id = [](const auto& id_pair) {
        return utility::element<0>(id_pair) != utility::element<1>(id_pair);
    };

    const auto func = [&mt, &ptr](const auto& id_pair) {
        const auto& [source_id, target_id] = id_pair;
        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        const auto ls = PlasticLocalSynapse{ target_id, source_id, weight };

        ptr->add_synapse(ls);
    };

    ranges::for_each(
        ranges::views::cartesian_product(NeuronID::range(number_neurons), NeuronID::range(number_neurons))
            | ranges::views::filter(not_the_same_id),
        func);

    return ptr;
}

std::shared_ptr<NetworkGraph> NetworkGraphFactory::construct_network_graph(const NeuronID::value_type number_neurons, const mpiPP::MPIRank mpi_rank, const NeuronID::value_type number_connections_per_vertex, std::mt19937& mt) {
    auto ptr = std::make_shared<NetworkGraph>(mpi_rank);
    ptr->init(number_neurons);

    for (auto i = 0ULL; i < number_connections_per_vertex; i++) {
        const auto& source_ids = NeuronID::range(number_neurons);
        const auto& target_ids = RandomFactory::get_random_derangement(number_neurons, mt);

        for (auto j = 0ULL; j < number_neurons; j++) {
            const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
            const auto ls = PlasticLocalSynapse(NeuronID(false, target_ids[j]), source_ids[j], weight);
            ptr->add_synapse(ls);
        }
    }

    return ptr;
}

void NetworkGraphFactory::construct_dense_plastic_network(const std::shared_ptr<NetworkGraph>& network_graph, const std::span<const SignalType>& signal_types,
                                                          NeuronID::value_type number_neurons, NeuronID::value_type number_outgoing_connections_per_neuron, int number_ranks, const mpiPP::MPIRank my_rank, std::mt19937& mt) {
    for (auto local_neuron_id = 0ULL; local_neuron_id < number_neurons; local_neuron_id++) {
        auto target_neurons = std::unordered_set<NeuronID>{};
        for (auto i = 0ULL; i < number_outgoing_connections_per_neuron; i++) {
            auto target_neuron_id = NeuronID{};
            do {
                target_neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, NeuronID(local_neuron_id), mt);
            } while (target_neurons.contains(target_neuron_id));
            target_neurons.insert(target_neuron_id);
            const auto rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);
            const auto weight = signal_types[local_neuron_id] == SignalType::Excitatory ? 1 : -1;
            if (rank == my_rank) {
                network_graph->add_synapse(PlasticLocalSynapse(target_neuron_id, NeuronID(local_neuron_id), weight));
            } else {
                network_graph->add_synapse(PlasticDistantOutSynapse(RankNeuronId(rank, target_neuron_id), NeuronID(local_neuron_id), weight));
            }
        }
    }
}

std::shared_ptr<NetworkGraph> NetworkGraphFactory::construct_dense_network_graph(NeuronID::value_type number_neurons, mpiPP::MPIRank mpi_rank, NeuronID::value_type number_connections_per_vertex) {
    auto ptr = std::make_shared<NetworkGraph>(mpi_rank);
    ptr->init(number_neurons);

    for (auto neuron_id = NeuronID::value_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        for (auto offset = NeuronID::value_type{ 1 }; offset <= number_connections_per_vertex; offset++) {
            const auto target_id = (neuron_id + offset) % number_neurons;

            const auto weight = 1;
            const auto ls = PlasticLocalSynapse{ NeuronID(target_id), NeuronID(neuron_id), weight };
            ptr->add_synapse(ls);
        }
    }

    return ptr;
}
