/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticEquallyWeightedActivityInput.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"

#include "mpi-wrapper/MPIInfo.h"

#include <span>

void SynapticEquallyWeightedActivityInput::update_local_input(std::span<const FiredStatus> fired, std::span<double> input, const NeuronID first, const NeuronID last) {
    const auto& network_graph = get_network_graph();
    const auto number_neurons = fired.size();

    const auto first_id = first.get_neuron_id();
    const auto last_id = last.get_neuron_id();

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto fired_status = fired[neuron_id];
        if (fired_status == FiredStatus::Inactive) {
            continue;
        }

        const auto& [local_out_edges_plastic, local_out_edges_static] = network_graph->get_local_out_edges(neuron_id);
        for (const auto& [target_neuron_id, synapse_strength] : local_out_edges_static) {
            const auto target_id = target_neuron_id.get_neuron_id();
            if (target_id < first_id || target_id >= last_id) {
                continue;
            }

            input[target_id] += static_cast<double>(synapse_strength);
        }

        for (const auto& [target_neuron_id, synapse_strength] : local_out_edges_plastic) {
            const auto target_id = target_neuron_id.get_neuron_id();
            if (target_id < first_id || target_id >= last_id) {
                continue;
            }

            input[target_id] += synapse_strength;
        }
    }
}

void SynapticEquallyWeightedActivityInput::update_distant_input(std::span<const FiredStatus> /*fired*/, std::span<double> input, const NeuronID first, const NeuronID last) {
    if (mpiPP::MPIInfo::get_number_ranks() == 1) {
        return;
    }

    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronID::range(first, last)) {
        auto local_plastic_input = RelearnTypes::plastic_synapse_weight{ 0 };
        auto local_static_input = RelearnTypes::static_synapse_weight{ 0 };

        const auto id = neuron_id.get_neuron_id();

        const auto& [distant_in_edges_plastic, distant_in_edges_static] = network_graph->get_distant_in_edges(id);

        for (const auto& [key, synapse_strength] : distant_in_edges_static) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (!contains_id) {
                continue;
            }

            local_static_input += synapse_strength;
        }

        for (const auto& [key, synapse_strength] : distant_in_edges_plastic) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (!contains_id) {
                continue;
            }

            local_plastic_input += synapse_strength;
        }

        const auto local_input = static_cast<double>(local_plastic_input) + local_static_input;
        input[id] += local_input;
    }
}
