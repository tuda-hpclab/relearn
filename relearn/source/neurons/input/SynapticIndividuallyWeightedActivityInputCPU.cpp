/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticIndividuallyWeightedActivityInputCPU.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <cmath>
#include <span>
#include <utility>

void SynapticIndividuallyWeightedActivityInputCPU::update_local_input(const std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) {
    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
        const auto id = neuron_id.get_neuron_id();
        if (fired[id] == FiredStatus::Inactive) {
            continue;
        }

        const auto& [local_out_edges_plastic, local_out_edges_static] = network_graph->get_local_out_edges(id);

        for (const auto& [target_neuron_id, synapse_strength] : local_out_edges_static) {
            input[target_neuron_id.get_neuron_id()] += utility::cast<activity_type>(synapse_strength); // no synaptic plasticity used here (yet)
        }

        const auto local_rank = mpiPP::MPIInfo::get_my_rank();
        const auto neuron_id_with_rank = RankNeuronId{ local_rank, neuron_id };

        for (const auto& [target_neuron_id, synapse_strength] : local_out_edges_plastic) {
            const auto& synapse_weight_vector = get_weights(weight_map, target_neuron_id, neuron_id_with_rank);

            RelearnException::check(std::cmp_equal(synapse_weight_vector.size(), std::abs(synapse_strength)), "SynapticIndividuallyWeightedActivityInput::update_local_input: size of weight vector does not equal synapse_strength");

            // sum up all the weights
            auto sum_of_weights = SynapticIndividuallyWeightedActivityInputCPU::weight_type{ 0 };
            for (const auto& weight : synapse_weight_vector) {
                sum_of_weights += weight;
            }
            input[target_neuron_id.get_neuron_id()] += utility::cast<activity_type>(sum_of_weights);
        }
    }
}

void SynapticIndividuallyWeightedActivityInputCPU::update_distant_input(const std::span<const FiredStatus> /*fired*/, std::span<activity_type> input, NeuronID first, NeuronID last) {
    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
        auto local_plastic_input = SynapticIndividuallyWeightedActivityInputCPU::weight_type{ 0 };
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

            const auto& synapse_weight_vector = get_weights(weight_map, neuron_id, key);

            RelearnException::check(std::cmp_equal(synapse_weight_vector.size(), std::abs(synapse_strength)), "SynapticIndividuallyWeightedActivityInput::update_distant_input: size of weight vector does not equal synapse_strength");

            auto sum_of_weights = SynapticIndividuallyWeightedActivityInputCPU::weight_type{ 0 };
            for (const auto& weight : synapse_weight_vector) {
                sum_of_weights += weight;
            }
            local_plastic_input += sum_of_weights;
        }

        const auto local_input = local_plastic_input + local_static_input;
        input[id] += local_input;
    }
}

[[nodiscard]] const SynapticIndividuallyWeightedActivityInputCPU::weight_vector_type& SynapticIndividuallyWeightedActivityInputCPU::get_weights(const SynapticIndividuallyWeightedActivityInputCPU::weight_map_type& weight_map_to_use, const NeuronID target_neuron, const RankNeuronId source_neuron) const {
    const auto target_neuron_id = target_neuron.get_neuron_id();

    const auto pair = std::make_pair(target_neuron_id, source_neuron);

    return weight_map_to_use.at(pair);
}

[[nodiscard]] const SynapticIndividuallyWeightedActivityInputCPU::weight_vector_type& SynapticIndividuallyWeightedActivityInputCPU::get_weights(const SynapticIndividuallyWeightedActivityInputCPU::weight_map_type& weight_map_to_use, const NeuronID::value_type target_neuron, const RankNeuronId source_neuron) const {
    const auto pair = std::make_pair(target_neuron, source_neuron);

    return weight_map_to_use.at(pair);
}
