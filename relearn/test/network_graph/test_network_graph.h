#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "Types.h"

#include <map>

class NetworkGraph;

class NetworkGraphTest : public RelearnTest {
protected:
    template <typename T, typename synapse_weight>
    void erase_empty(std::map<T, synapse_weight>& edges) {
        std::erase_if(edges, [](const auto& val) { return val.second == 0; });
    }

    template <typename T, typename synapse_weight>
    void erase_empties(std::map<T, std::map<T, synapse_weight>>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            erase_empty<T>(iterator->second);

            if (iterator->second.empty()) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    static void assert_local_plastic_empty(const NetworkGraph& network_graph);

    static void assert_distant_plastic_empty(const NetworkGraph& network_graph);

    static void assert_plastic_empty(const NetworkGraph& network_graph);

    static void assert_local_static_empty(const NetworkGraph& network_graph);

    static void assert_distant_static_empty(const NetworkGraph& network_graph);

    static void assert_static_empty(const NetworkGraph& network_graph);

    static void assert_plastic_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons);

    static void assert_static_size(const NetworkGraph& network_graph, RelearnTypes::number_neurons_type expected_number_neurons);

    /**
     * @brief Format is: target -> source -> weight
     */
    static void assert_in_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>& incoming_edges,
        const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>>& distant_incoming_edges);

    /**
     * @brief Format is: source -> target-> weight
     */
    static void assert_out_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::plastic_synapse_weight>>& outgoing_edges,
        const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::plastic_synapse_weight>>& distant_outgoing_edges);

    /**
     * @brief Format is: target -> source -> weight
     */
    static void assert_in_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>& incoming_edges,
        const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>>& distant_incoming_edges);

    /**
     * @brief Format is: source -> target-> weight
     */
    static void assert_out_connectivity(const NetworkGraph& network_graph, const std::map<NeuronID, std::map<NeuronID, RelearnTypes::static_synapse_weight>>& outgoing_edges,
        const std::map<NeuronID, std::map<RankNeuronId, RelearnTypes::static_synapse_weight>>& distant_outgoing_edges);
};
