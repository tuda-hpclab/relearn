/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_network_graph.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/BasicTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include "adapter/network_graph/NetworkGraphAdapter.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

void NetworkGraphTest::assert_local_plastic_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    const auto& [all_local_in_edges, _1] = network_graph.get_all_local_in_edges();
    const auto& [all_local_out_edges, _2] = network_graph.get_all_local_out_edges();

    for (const auto& neighborhood : all_local_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_local_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& [local_in_edges, _3] = network_graph.get_local_in_edges(neuron_id);
        const auto& [_all_local_out_edges, _4] = network_graph.get_local_out_edges(neuron_id);

        ASSERT_TRUE(local_in_edges.empty());
        ASSERT_TRUE(_all_local_out_edges.empty());
    }
}

void NetworkGraphTest::assert_distant_plastic_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    const auto& [all_distant_in_edges, _1] = network_graph.get_all_distant_in_edges();
    const auto& [all_distant_out_edges, _2] = network_graph.get_all_distant_out_edges();

    for (const auto& neighborhood : all_distant_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_distant_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& [distant_in_edges, _3] = network_graph.get_distant_in_edges(neuron_id);
        const auto& [distant_out_edges, _4] = network_graph.get_distant_out_edges(neuron_id);

        ASSERT_TRUE(distant_in_edges.empty());
        ASSERT_TRUE(distant_out_edges.empty());

#ifndef RELEARN_CUDA_ENABLED
        // get_all_plastic_partners_incoming/outgoing are unconditionally "Not supported on cuda"
        // (NetworkGraph.h's GPU-backed NetworkGraphBase doesn't expose host-side all-partners
        // queries), so this part of the shared helper only applies to CPU builds.
        const auto& incoming_excitatory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory);
        for (const auto& [rank, _] : incoming_excitatory_partners) {
            ASSERT_EQ(rank, network_graph.get_mpi_rank());
        }

        const auto& incoming_inhibitory_partners = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory);
        for (const auto& [rank, _] : incoming_inhibitory_partners) {
            ASSERT_EQ(rank, network_graph.get_mpi_rank());
        }

        const auto& outgoing_partners = network_graph.get_all_plastic_partners_outgoing(neuron_id);
        for (const auto& [rank, _] : outgoing_partners) {
            ASSERT_EQ(rank, network_graph.get_mpi_rank());
        }
#endif // RELEARN_CUDA_ENABLED
    }
}

void NetworkGraphTest::assert_plastic_empty(const NetworkGraph& network_graph) {
    assert_local_plastic_empty(network_graph);
    assert_distant_plastic_empty(network_graph);

    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto [number_excitatory_in_edges, _1] = network_graph.get_number_excitatory_in_edges(neuron_id);
        const auto [number_inhibitiry_in_edges, _2] = network_graph.get_number_inhibitory_in_edges(neuron_id);
        const auto [number_out_edges, _3] = network_graph.get_number_out_edges(neuron_id);

        ASSERT_EQ(number_excitatory_in_edges, 0);
        ASSERT_EQ(number_inhibitiry_in_edges, 0);
        ASSERT_EQ(number_out_edges, 0);
    }
}

void NetworkGraphTest::assert_local_static_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    const auto& [_1, all_local_in_edges] = network_graph.get_all_local_in_edges();
    const auto& [_2, all_local_out_edges] = network_graph.get_all_local_out_edges();

    for (const auto& neighborhood : all_local_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_local_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& [_3, local_in_edges] = network_graph.get_local_in_edges(neuron_id);
        const auto& [_4, _all_local_out_edges] = network_graph.get_local_out_edges(neuron_id);

        ASSERT_TRUE(local_in_edges.empty());
        ASSERT_TRUE(_all_local_out_edges.empty());
    }
}

void NetworkGraphTest::assert_distant_static_empty(const NetworkGraph& network_graph) {
    const auto number_neurons = network_graph.get_number_neurons();

    const auto& [_1, all_distant_in_edges] = network_graph.get_all_distant_in_edges();
    const auto& [_2, all_distant_out_edges] = network_graph.get_all_distant_out_edges();

    for (const auto& neighborhood : all_distant_in_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto& neighborhood : all_distant_out_edges) {
        ASSERT_TRUE(neighborhood.empty());
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& [_3, distant_in_edges] = network_graph.get_distant_in_edges(neuron_id);
        const auto& [_4, distant_out_edges] = network_graph.get_distant_out_edges(neuron_id);

        ASSERT_TRUE(distant_in_edges.empty());
        ASSERT_TRUE(distant_out_edges.empty());
    }
}

void NetworkGraphTest::assert_static_empty(const NetworkGraph& network_graph) {
    assert_local_static_empty(network_graph);
    assert_distant_static_empty(network_graph);

    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto [_1, number_excitatory_in_edges] = network_graph.get_number_excitatory_in_edges(neuron_id);
        const auto [_2, number_inhibitiry_in_edges] = network_graph.get_number_inhibitory_in_edges(neuron_id);
        const auto [_3, number_out_edges] = network_graph.get_number_out_edges(neuron_id);

        ASSERT_EQ(number_excitatory_in_edges, 0);
        ASSERT_EQ(number_inhibitiry_in_edges, 0);
        ASSERT_EQ(number_out_edges, 0);
    }
}

void NetworkGraphTest::assert_plastic_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons) {
    const auto number_neurons = network_graph.get_number_neurons();
    ASSERT_EQ(number_neurons, expected_number_neurons);

    const auto& [all_distant_in_edges, _1] = network_graph.get_all_distant_in_edges();
    const auto& [all_distant_out_edges, _2] = network_graph.get_all_distant_out_edges();
    const auto& [all_local_in_edges, _3] = network_graph.get_all_local_in_edges();
    const auto& [all_local_out_edges, _4] = network_graph.get_all_local_out_edges();

    ASSERT_EQ(expected_number_neurons, all_distant_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_distant_out_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_out_edges.size());

#ifndef RELEARN_CUDA_ENABLED
    // get_all_plastic_partners_incoming/outgoing are unconditionally "Not supported on cuda".
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_NO_THROW(std::ignore = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory););
        ASSERT_NO_THROW(std::ignore = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory););
        ASSERT_NO_THROW(std::ignore = network_graph.get_all_plastic_partners_outgoing(neuron_id););
    }
#endif // RELEARN_CUDA_ENABLED

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons, number_neurons + number_neurons_out_of_scope)) {
        ASSERT_THROW_NO_PRINT(std::ignore = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Excitatory);, RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = network_graph.get_all_plastic_partners_incoming(neuron_id, SignalType::Inhibitory);, RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = network_graph.get_all_plastic_partners_outgoing(neuron_id);, RelearnException);
    }
}

void NetworkGraphTest::assert_static_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons) {
    const auto number_neurons = network_graph.get_number_neurons();
    ASSERT_EQ(number_neurons, expected_number_neurons);

    const auto& [_1, all_distant_in_edges] = network_graph.get_all_distant_in_edges();
    const auto& [_2, all_distant_out_edges] = network_graph.get_all_distant_out_edges();
    const auto& [_3, all_local_in_edges] = network_graph.get_all_local_in_edges();
    const auto& [_4, all_local_out_edges] = network_graph.get_all_local_out_edges();

    ASSERT_EQ(expected_number_neurons, all_distant_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_distant_out_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_in_edges.size());
    ASSERT_EQ(expected_number_neurons, all_local_out_edges.size());
}

void NetworkGraphTest::assert_in_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>& incoming_edges,
                                              const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>>& distant_incoming_edges) {
    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& golden_in_edges = incoming_edges.contains(neuron_id) ? incoming_edges.at(neuron_id) : std::map<NeuronID, RelearnTypes::plastic_synapse_weight>{};

        const auto& [local_in_edges, _1] = network_graph.get_local_in_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(local_in_edges.size(), golden_in_edges.size());

        for (const auto& [other_neuron_id, weight] : local_in_edges) {
            ASSERT_TRUE(golden_in_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_in_edges.at(other_neuron_id));
        }

        const auto& [all_local_in_edges, _2] = network_graph.get_all_local_in_edges();
        ASSERT_EQ(all_local_in_edges[neuron_id.get_neuron_id()], local_in_edges);

        const auto& golden_distant_in_edges = distant_incoming_edges.contains(neuron_id) ? distant_incoming_edges.at(neuron_id) : std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>{};

        const auto& [distant_in_edges, _3] = network_graph.get_distant_in_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(golden_distant_in_edges.size(), golden_distant_in_edges.size());

        for (const auto& [other_neuron_id, weight] : distant_in_edges) {
            ASSERT_TRUE(golden_distant_in_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_distant_in_edges.at(other_neuron_id));
        }

        const auto& [all_distant_in_edges, _4] = network_graph.get_all_distant_in_edges();
        ASSERT_EQ(all_distant_in_edges[neuron_id.get_neuron_id()], distant_in_edges);

        const auto [number_excitatory, _5] = network_graph.get_number_excitatory_in_edges(neuron_id.get_neuron_id());
        const auto [number_inhibitory, _6] = network_graph.get_number_inhibitory_in_edges(neuron_id.get_neuron_id());

        const auto excitatory_summer = [](const auto previous, const auto& p) {
            if (p.second < 0) {
                return previous;
            }
            return previous + p.second;
        };

        const auto inhibitory_summer = [](const auto previous, const auto& p) {
            if (p.second > 0) {
                return previous;
            }
            return previous + std::abs(p.second);
        };

        const auto golden_excitatory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                                                                      RelearnTypes::plastic_synapse_weight{ 0 }, excitatory_summer);

        const auto golden_inhibitory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                                                                      RelearnTypes::plastic_synapse_weight{ 0 }, inhibitory_summer);

        const auto golden_excitatory_distant_in_edges_count = std::accumulate(golden_distant_in_edges.cbegin(), golden_distant_in_edges.cend(),
                                                                              RelearnTypes::plastic_synapse_weight{ 0 }, excitatory_summer);

        const auto golden_inhibitory_distant_in_edges_count = std::accumulate(golden_distant_in_edges.cbegin(), golden_distant_in_edges.cend(),
                                                                              RelearnTypes::plastic_synapse_weight{ 0 }, inhibitory_summer);

        ASSERT_EQ(golden_excitatory_distant_in_edges_count + golden_excitatory_in_edges_count, number_excitatory);
        ASSERT_EQ(golden_inhibitory_distant_in_edges_count + golden_inhibitory_in_edges_count, number_inhibitory);
    }
}

void NetworkGraphTest::assert_out_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>& outgoing_edges,
                                               const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>>& distant_outgoing_edges) {
    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& golden_out_edges = outgoing_edges.contains(neuron_id) ? outgoing_edges.at(neuron_id) : std::map<NeuronID, RelearnTypes::plastic_synapse_weight>{};

        const auto& [local_out_edges, _1] = network_graph.get_local_out_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(local_out_edges.size(), golden_out_edges.size());

        for (const auto& [other_neuron_id, weight] : local_out_edges) {
            ASSERT_TRUE(golden_out_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_out_edges.at(other_neuron_id));
        }

        const auto& [all_local_out_edges, _2] = network_graph.get_all_local_out_edges();
        ASSERT_EQ(all_local_out_edges[neuron_id.get_neuron_id()], local_out_edges);

        const auto& golden_distant_out_edges = distant_outgoing_edges.contains(neuron_id) ? distant_outgoing_edges.at(neuron_id) : std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>{};

        const auto& [distant_out_edges, _3] = network_graph.get_distant_out_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(golden_distant_out_edges.size(), golden_distant_out_edges.size());

        for (const auto& [other_neuron_id, weight] : distant_out_edges) {
            ASSERT_TRUE(golden_distant_out_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_distant_out_edges.at(other_neuron_id));
        }

        const auto& [all_distant_out_edges, _4] = network_graph.get_all_distant_out_edges();
        ASSERT_EQ(all_distant_out_edges[neuron_id.get_neuron_id()], distant_out_edges);

        const auto [number_edges, _5] = network_graph.get_number_out_edges(neuron_id.get_neuron_id());

        const auto summer = [](const auto previous, const auto& p) {
            return previous + std::abs(p.second);
        };

        const auto golden_excitatory_out_edges_count = std::accumulate(golden_out_edges.cbegin(), golden_out_edges.cend(),
                                                                       RelearnTypes::plastic_synapse_weight{ 0 }, summer);

        const auto golden_excitatory_distant_out_edges_count = std::accumulate(golden_distant_out_edges.cbegin(), golden_distant_out_edges.cend(),
                                                                               RelearnTypes::plastic_synapse_weight{ 0 }, summer);

        ASSERT_EQ(golden_excitatory_distant_out_edges_count + golden_excitatory_out_edges_count, number_edges);
    }
}

void NetworkGraphTest::assert_in_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>& incoming_edges,
                                              const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>>& distant_incoming_edges) {
    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& golden_in_edges = incoming_edges.contains(neuron_id) ? incoming_edges.at(neuron_id) : std::map<NeuronID, RelearnTypes::static_synapse_weight>{};

        const auto& [_1, local_in_edges] = network_graph.get_local_in_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(local_in_edges.size(), golden_in_edges.size());

        for (const auto& [other_neuron_id, weight] : local_in_edges) {
            ASSERT_TRUE(golden_in_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_in_edges.at(other_neuron_id));
        }

        const auto& [_2, all_local_in_edges] = network_graph.get_all_local_in_edges();
        ASSERT_EQ(all_local_in_edges[neuron_id.get_neuron_id()], local_in_edges);

        const auto& golden_distant_in_edges = distant_incoming_edges.contains(neuron_id) ? distant_incoming_edges.at(neuron_id) : std::map<RankNeuronId, RelearnTypes::static_synapse_weight>{};

        const auto& [_3, distant_in_edges] = network_graph.get_distant_in_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(golden_distant_in_edges.size(), golden_distant_in_edges.size());

        for (const auto& [other_neuron_id, weight] : distant_in_edges) {
            ASSERT_TRUE(golden_distant_in_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_distant_in_edges.at(other_neuron_id));
        }

        const auto& [_4, all_distant_in_edges] = network_graph.get_all_distant_in_edges();
        ASSERT_EQ(all_distant_in_edges[neuron_id.get_neuron_id()], distant_in_edges);

        const auto [_5, number_excitatory] = network_graph.get_number_excitatory_in_edges(neuron_id.get_neuron_id());
        const auto [_6, number_inhibitory] = network_graph.get_number_inhibitory_in_edges(neuron_id.get_neuron_id());

        const auto excitatory_summer = [](const auto previous, const auto& p) {
            if (p.second < 0) {
                return previous;
            }
            return previous + p.second;
        };

        const auto inhibitory_summer = [](const auto previous, const auto& p) {
            if (p.second > 0) {
                return previous;
            }
            return previous + std::abs(p.second);
        };

        const auto golden_excitatory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                                                                      RelearnTypes::static_synapse_weight{ 0 }, excitatory_summer);

        const auto golden_inhibitory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                                                                      RelearnTypes::static_synapse_weight{ 0 }, inhibitory_summer);

        const auto golden_excitatory_distant_in_edges_count = std::accumulate(golden_distant_in_edges.cbegin(), golden_distant_in_edges.cend(),
                                                                              RelearnTypes::static_synapse_weight{ 0 }, excitatory_summer);

        const auto golden_inhibitory_distant_in_edges_count = std::accumulate(golden_distant_in_edges.cbegin(), golden_distant_in_edges.cend(),
                                                                              RelearnTypes::static_synapse_weight{ 0 }, inhibitory_summer);

        ASSERT_NEAR(golden_excitatory_distant_in_edges_count + golden_excitatory_in_edges_count, number_excitatory, eps);
        ASSERT_NEAR(golden_inhibitory_distant_in_edges_count + golden_inhibitory_in_edges_count, number_inhibitory, eps);
    }
}

void NetworkGraphTest::assert_out_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>& outgoing_edges,
                                               const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>>& distant_outgoing_edges) {
    const auto number_neurons = network_graph.get_number_neurons();

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& golden_out_edges = outgoing_edges.contains(neuron_id) ? outgoing_edges.at(neuron_id) : std::map<NeuronID, RelearnTypes::static_synapse_weight>{};

        const auto& [_1, local_out_edges] = network_graph.get_local_out_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(local_out_edges.size(), golden_out_edges.size());

        for (const auto& [other_neuron_id, weight] : local_out_edges) {
            ASSERT_TRUE(golden_out_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_out_edges.at(other_neuron_id));
        }

        const auto& [_2, all_local_out_edges] = network_graph.get_all_local_out_edges();
        ASSERT_EQ(all_local_out_edges[neuron_id.get_neuron_id()], local_out_edges);

        const auto& golden_distant_out_edges = distant_outgoing_edges.contains(neuron_id) ? distant_outgoing_edges.at(neuron_id) : std::map<RankNeuronId, RelearnTypes::static_synapse_weight>{};

        const auto& [_3, distant_out_edges] = network_graph.get_distant_out_edges(neuron_id.get_neuron_id());
        ASSERT_EQ(golden_distant_out_edges.size(), golden_distant_out_edges.size());

        for (const auto& [other_neuron_id, weight] : distant_out_edges) {
            ASSERT_TRUE(golden_distant_out_edges.contains(other_neuron_id));
            ASSERT_EQ(weight, golden_distant_out_edges.at(other_neuron_id));
        }

        const auto& [_4, all_distant_out_edges] = network_graph.get_all_distant_out_edges();
        ASSERT_EQ(all_distant_out_edges[neuron_id.get_neuron_id()], distant_out_edges);

        const auto [_5, number_edges] = network_graph.get_number_out_edges(neuron_id.get_neuron_id());

        const auto summer = [](const auto previous, const auto& p) {
            return previous + std::abs(p.second);
        };

        const auto golden_excitatory_out_edges_count = std::accumulate(golden_out_edges.cbegin(), golden_out_edges.cend(),
                                                                       RelearnTypes::static_synapse_weight{ 0 }, summer);

        const auto golden_excitatory_distant_out_edges_count = std::accumulate(golden_distant_out_edges.cbegin(), golden_distant_out_edges.cend(),
                                                                               RelearnTypes::static_synapse_weight{ 0 }, summer);

        ASSERT_NEAR(golden_excitatory_distant_out_edges_count + golden_excitatory_out_edges_count, number_edges, eps);
    }
}

TEST_F(NetworkGraphTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);

    const auto network_graph = NetworkGraph(rank);

    ASSERT_EQ(network_graph.get_number_neurons(), 0);
    ASSERT_EQ(network_graph.get_mpi_rank(), rank);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);
}

TEST_F(NetworkGraphTest, testConstructorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(NetworkGraph ng_exception(mpiPP::MPIRank::uninitialized_rank());, RelearnException);
}

TEST_F(NetworkGraphTest, testInit) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(rank);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ASSERT_EQ(network_graph.get_mpi_rank(), rank);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons);
    assert_static_size(network_graph, number_neurons);
}

TEST_F(NetworkGraphTest, testInitException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(rank);

    ASSERT_THROW_NO_PRINT(network_graph.init(0, NetworkGPUType::MEMORY_POOL), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, 0);
    assert_static_size(network_graph, 0);

    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ASSERT_THROW_NO_PRINT(network_graph.init(0, NetworkGPUType::MEMORY_POOL), RelearnException);
    ASSERT_THROW_NO_PRINT(network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons);
    assert_static_size(network_graph, number_neurons);

    ASSERT_EQ(network_graph.get_mpi_rank(), rank);
}

#ifndef RELEARN_CUDA_ENABLED
TEST_F(NetworkGraphTest, testCreateNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(mt);

    const auto number_neurons = number_neurons_init + number_neurons_create;

    auto network_graph = NetworkGraph(rank);
    network_graph.init(number_neurons_init, NetworkGPUType::MEMORY_POOL);
    network_graph.create_neurons(number_neurons_create);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons);
    assert_static_size(network_graph, number_neurons);

    ASSERT_EQ(network_graph.get_mpi_rank(), rank);
}

TEST_F(NetworkGraphTest, testCreateNeuronsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(mt);

    const auto number_neurons = number_neurons_init + number_neurons_create;

    auto network_graph = NetworkGraph(rank);

    ASSERT_THROW_NO_PRINT(network_graph.create_neurons(0), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, 0);
    assert_static_size(network_graph, 0);

    ASSERT_THROW_NO_PRINT(network_graph.create_neurons(number_neurons_create), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, 0);
    assert_static_size(network_graph, 0);

    ASSERT_THROW_NO_PRINT(network_graph.create_neurons(0), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, 0);
    assert_static_size(network_graph, 0);

    network_graph.init(number_neurons_init, NetworkGPUType::MEMORY_POOL);

    ASSERT_THROW_NO_PRINT(network_graph.create_neurons(0), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons_init);
    assert_static_size(network_graph, number_neurons_init);

    network_graph.create_neurons(number_neurons_create);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons);
    assert_static_size(network_graph, number_neurons);

    ASSERT_THROW_NO_PRINT(network_graph.create_neurons(0), RelearnException);

    assert_plastic_empty(network_graph);
    assert_static_empty(network_graph);

    assert_plastic_size(network_graph, number_neurons);
    assert_static_size(network_graph, number_neurons);

    ASSERT_EQ(network_graph.get_mpi_rank(), rank);
}
#endif

TEST_F(NetworkGraphTest, testPlasticLocalSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    assert_static_empty(network_graph);
    assert_distant_plastic_empty(network_graph);

    assert_in_connectivity(network_graph, incoming_edges, {});
    assert_out_connectivity(network_graph, outgoing_edges, {});
}

TEST_F(NetworkGraphTest, testPlasticDistantInSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    assert_static_empty(network_graph);
    assert_local_plastic_empty(network_graph);

    assert_out_connectivity(network_graph, std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>{}, {});
    assert_in_connectivity(network_graph, {}, incoming_edges);
}

TEST_F(NetworkGraphTest, testPlasticDistantOutSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto distant_out_synapses = SynapsesFactory::generate_plastic_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_static_empty(network_graph);
    assert_local_plastic_empty(network_graph);

    assert_out_connectivity(network_graph, {}, outgoing_edges);
    assert_in_connectivity(network_graph, std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>{}, {});
}

TEST_F(NetworkGraphTest, testStaticLocalSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_static_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    assert_plastic_empty(network_graph);
    assert_distant_static_empty(network_graph);

    assert_in_connectivity(network_graph, incoming_edges, {});
    assert_out_connectivity(network_graph, outgoing_edges, {});
}

TEST_F(NetworkGraphTest, testStaticDistantInSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto distant_in_synapses = SynapsesFactory::generate_static_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    assert_plastic_empty(network_graph);
    assert_local_static_empty(network_graph);

    assert_out_connectivity(network_graph, std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>{}, {});
    assert_in_connectivity(network_graph, {}, incoming_edges);
}

TEST_F(NetworkGraphTest, testStaticDistantOutSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto distant_out_synapses = SynapsesFactory::generate_static_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_plastic_empty(network_graph);
    assert_local_static_empty(network_graph);

    assert_out_connectivity(network_graph, {}, outgoing_edges);
    assert_in_connectivity(network_graph, std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>{}, {});
}

TEST_F(NetworkGraphTest, testPlasticAddSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_plastic_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_static_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testPlasticAddSynapsesException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticLocalSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticLocalSynapse(NeuronID(target.get_neuron_id() + number_neurons), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticLocalSynapse(target, NeuronID(source.get_neuron_id() + number_neurons), weight)), RelearnException);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantInSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantInSynapse(NeuronID(target.get_neuron_id() + number_neurons), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantInSynapse(target, RankNeuronId(mpiPP::MPIRank::root_rank(), source.get_neuron_id()), weight)), RelearnException);
    }

    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_plastic_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantOutSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantOutSynapse(RankNeuronId(mpiPP::MPIRank::root_rank(), target.get_neuron_id()), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticDistantOutSynapse(target, NeuronID(source.get_neuron_id() + number_neurons), weight)), RelearnException);
    }

    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_static_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testPlasticAddSynapseSelfLoopException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
    const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

    ASSERT_THROW_NO_PRINT(network_graph.add_synapse(PlasticLocalSynapse(neuron_id, neuron_id, weight)), RelearnException);
}

TEST_F(NetworkGraphTest, testStaticAddSynapseSelfLoopException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
    const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

    ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticLocalSynapse(neuron_id, neuron_id, weight)), RelearnException);
}

TEST_F(NetworkGraphTest, testPlasticAddEdges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_plastic_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    network_graph.add_edges(local_synapses, distant_in_synapses, distant_out_synapses);

    assert_static_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testPlasticAddEdgesSelfLoopException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
    const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);

    const auto local_synapses = PlasticLocalSynapses{ PlasticLocalSynapse(neuron_id, neuron_id, weight) };

    ASSERT_THROW_NO_PRINT(network_graph.add_edges(local_synapses, PlasticDistantInSynapses{}, PlasticDistantOutSynapses{}), RelearnException);
}

TEST_F(NetworkGraphTest, testStaticAddEdgesSelfLoopException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
    const auto weight = SynapsesFactory::get_random_static_synapse_weight(mt);

    const auto local_synapses = StaticLocalSynapses{ StaticLocalSynapse(neuron_id, neuron_id, weight) };

    ASSERT_THROW_NO_PRINT(network_graph.add_edges(local_synapses, StaticDistantInSynapses{}, StaticDistantOutSynapses{}), RelearnException);
}

TEST_F(NetworkGraphTest, testStaticAddSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_static_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_static_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_static_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);
    }

    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_plastic_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testStaticAddSynapsesException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_static_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticLocalSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticLocalSynapse(NeuronID(target.get_neuron_id() + number_neurons), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticLocalSynapse(target, NeuronID(source.get_neuron_id() + number_neurons), weight)), RelearnException);
    }

    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_static_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantInSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantInSynapse(NeuronID(target.get_neuron_id() + number_neurons), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantInSynapse(target, RankNeuronId(mpiPP::MPIRank::root_rank(), source.get_neuron_id()), weight)), RelearnException);
    }

    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_static_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);

        const auto& [target, source, weight] = synapse;
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantOutSynapse(target, source, 0)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantOutSynapse(RankNeuronId(mpiPP::MPIRank::root_rank(), target.get_neuron_id()), source, weight)), RelearnException);
        ASSERT_THROW_NO_PRINT(network_graph.add_synapse(StaticDistantOutSynapse(target, NeuronID(source.get_neuron_id() + number_neurons), weight)), RelearnException);
    }

    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    assert_plastic_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testStaticAddEdges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_static_local_synapses(number_neurons, number_synapses, mt);
    const auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_static_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    const auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_static_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    const auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    network_graph.add_edges(local_synapses, distant_in_synapses, distant_out_synapses);

    assert_plastic_empty(network_graph);

    assert_out_connectivity(network_graph, outgoing_edges, distant_outgoing_edges);
    assert_in_connectivity(network_graph, incoming_edges, distant_incoming_edges);
}

TEST_F(NetworkGraphTest, testSynapsesRemoval) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank());
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    for (const auto& [target, source, weight] : local_synapses) {
        network_graph.add_synapse(PlasticLocalSynapse(target, source, -weight));
    }

    assert_static_empty(network_graph);
    assert_plastic_empty(network_graph);
}

#ifndef RELEARN_CUDA_ENABLED
// get_all_plastic_partners_incoming/outgoing are unconditionally "Not supported on cuda" (see
// NetworkGraph.h's NetworkGraphBase::get_all_partners_incoming/outgoing) -- this whole test only
// applies to CPU builds.
TEST_F(NetworkGraphTest, testPlasticPartners) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt) + 1;

    auto network_graph = NetworkGraph(mpiPP::MPIRank::root_rank(), number_ranks);
    network_graph.init(number_neurons, NetworkGPUType::MEMORY_POOL);

    const auto add_static_stuff = [&]() {
        const auto local_synapses = SynapsesFactory::generate_static_local_synapses(number_neurons, number_synapses, mt);
        for (const auto& synapse : local_synapses) {
            network_graph.add_synapse(synapse);
        }

        const auto distant_in_synapses = SynapsesFactory::generate_static_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
        for (const auto& synapse : distant_in_synapses) {
            network_graph.add_synapse(synapse);
        }

        const auto distant_out_synapses = SynapsesFactory::generate_static_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
        for (const auto& synapse : distant_out_synapses) {
            network_graph.add_synapse(synapse);
        }
    };
    add_static_stuff();

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons, number_synapses, mt);
    for (const auto& synapse : local_synapses) {
        network_graph.add_synapse(synapse);
    }

    auto [incoming_edges, outgoing_edges] = NetworkGraphAdapter::transform_synapses(local_synapses);

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_in_synapses) {
        network_graph.add_synapse(synapse);
    }

    auto distant_incoming_edges = NetworkGraphAdapter::transform_synapses(distant_in_synapses);

    const auto distant_out_synapses = SynapsesFactory::generate_plastic_distant_out_synapses(number_neurons, number_synapses, number_ranks, number_foreign_neurons, mt);
    for (const auto& synapse : distant_out_synapses) {
        network_graph.add_synapse(synapse);
    }

    auto distant_outgoing_edges = NetworkGraphAdapter::transform_synapses(distant_out_synapses);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& expected_incoming = distant_incoming_edges[neuron_id];

        auto expected_incoming_inhibitory = std::unordered_set<RankNeuronId>{};
        for (const auto& [rni, weight] : expected_incoming) {
            if (weight < 0) {
                expected_incoming_inhibitory.emplace(rni);
            }
        }
        for (const auto& [id, weight] : incoming_edges[neuron_id]) {
            if (weight < 0) {
                expected_incoming_inhibitory.emplace(mpiPP::MPIRank::root_rank(), id);
            }
        }

        auto expected_incoming_excitatory = std::unordered_set<RankNeuronId>{};
        for (const auto& [rni, weight] : expected_incoming) {
            if (weight > 0) {
                expected_incoming_excitatory.emplace(rni);
            }
        }
        for (const auto& [id, weight] : incoming_edges[neuron_id]) {
            if (weight > 0) {
                expected_incoming_excitatory.emplace(mpiPP::MPIRank::root_rank(), id);
            }
        }

        const auto& actual_incoming_inhibitory = network_graph.get_all_plastic_partners_incoming(neuron_id.get_neuron_id(), SignalType::Inhibitory);
        const auto& actual_incoming_excitatory = network_graph.get_all_plastic_partners_incoming(neuron_id.get_neuron_id(), SignalType::Excitatory);

        ASSERT_EQ(actual_incoming_inhibitory, expected_incoming_inhibitory);
        ASSERT_EQ(actual_incoming_excitatory, expected_incoming_excitatory);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons)) {
        const auto& expected_outgoing = distant_outgoing_edges[neuron_id];
        auto expected_outgoing_unordered = std::unordered_set<RankNeuronId>{};
        std::ranges::transform(expected_outgoing, std::inserter(expected_outgoing_unordered, expected_outgoing_unordered.begin()),
                               [](const auto val) -> RankNeuronId { const auto [rni, _] = val; return rni; });

        std::transform(outgoing_edges[neuron_id].begin(), outgoing_edges[neuron_id].end(), std::inserter(expected_outgoing_unordered, expected_outgoing_unordered.begin()),
                       [](const auto val) -> RankNeuronId { const auto [id, _] = val; return RankNeuronId(mpiPP::MPIRank::root_rank(), id); });

        const auto& actual_incoming = network_graph.get_all_plastic_partners_outgoing(neuron_id.get_neuron_id());

        ASSERT_EQ(actual_incoming, expected_outgoing_unordered);
    }
}
#endif // RELEARN_CUDA_ENABLED
