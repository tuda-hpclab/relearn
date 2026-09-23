#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NetworkGraph.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <memory>
#include <random>
#include <vector>

class NetworkGraphFactory {
public:
    static std::shared_ptr<NetworkGraph> construct_empty_network_graph(RelearnTypes::number_neurons_type number_neurons, mpiPP::MPIRank mpi_rank = mpiPP::MPIRank::root_rank(), int number_ranks = 1);

    static std::shared_ptr<NetworkGraph> construct_all_to_all_network_graph(RelearnTypes::number_neurons_type number_neurons, mpiPP::MPIRank mpi_rank = mpiPP::MPIRank::root_rank());

    static std::shared_ptr<NetworkGraph> construct_all_to_all_network_graph(RelearnTypes::number_neurons_type number_neurons, mpiPP::MPIRank mpi_rank, std::mt19937& mt);

    static std::shared_ptr<NetworkGraph> construct_network_graph(RelearnTypes::number_neurons_type number_neurons, mpiPP::MPIRank mpi_rank, RelearnTypes::number_synapse_type number_connections_per_vertex, std::mt19937& mt);

    /**
     * @brief Adds for each neurons the specified number of outgoing connections to the network graph on a single rank
     * @param network_graph The network graph to which is modified for the current rank
     * @param signal_types Vector of signl_types on current rank
     * @param number_neurons Number of neurons per rank
     * @param number_outgoing_connections_per_neuron Number of outgoing connection that shall be added to each neuron
     * @param number_ranks Number of neurons per rank
     * @param my_rank Current (simulated) mpi rank
     * @param mt seed
     */
    static void construct_dense_plastic_network(const std::shared_ptr<NetworkGraph>& network_graph, const std::span<const SignalType>& signal_types, RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_synapse_type number_outgoing_connections_per_neuron, int number_ranks, const mpiPP::MPIRank my_rank, std::mt19937& mt);
    
    static std::shared_ptr<NetworkGraph> construct_dense_network_graph(RelearnTypes::number_neurons_type number_neurons, mpiPP::MPIRank mpi_rank, RelearnTypes::number_synapse_type number_connections_per_vertex);

    static std::vector<std::shared_ptr<NetworkGraph>> construct_multi_rank_network_graph(RelearnTypes::number_neurons_type number_neurons, const std::size_t number_ranks, std::size_t number_local_edges_per_neuron, std::size_t number_distant_edges_per_neuron, std::mt19937& mt);
};
