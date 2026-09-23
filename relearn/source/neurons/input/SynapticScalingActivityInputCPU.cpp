/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticScalingActivityInputCPU.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cmath>
#include <span>

void SynapticScalingActivityInputCPU::ensure_scales() {
    const auto scales_size = scales.size();
    RelearnException::check(scales_size > 0, "SynapticScalingActivityInput::ensure_scales: There are no scales.");

    const auto last_value = scales[scales_size - 1];
    if (last_value != activity_type{ 0 }) {
        return;
    }

    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronIDRange::range(scales_size)) {
        const auto id = neuron_id.get_neuron_id();
        if (scales[id] != activity_type{ 0 }) {
            continue;
        }

        const auto& [plastic_excitatory_in_edges, _1] = network_graph->get_number_excitatory_in_edges(id);
        const auto& [plastic_inhibitory_in_edges, _2] = network_graph->get_number_inhibitory_in_edges(id);

        const auto s1 = std::abs(static_cast<activity_type>(plastic_excitatory_in_edges));
        const auto s2 = std::abs(static_cast<activity_type>(plastic_inhibitory_in_edges));

        const auto sum = s1 + s2;

        scales[id] = t_scale / (sum == activity_type{ 0 } ? activity_type{ 1 } : sum);
    }

    needs_scale_update = false;
}

void SynapticScalingActivityInputCPU::update_local_input(std::span<const FiredStatus> fired, std::span<activity_type> input, const NeuronID first, const NeuronID last) {
    if (needs_scale_update) {
        ensure_scales();
    }

    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
        const auto id = neuron_id.get_neuron_id();

        const auto scale = scales[id];
        const auto& [local_in_edges_plastic, local_in_edges_static] = network_graph->get_local_in_edges(id);

        for (const auto& [src_neuron_id, synapse_strength] : local_in_edges_static) {
            if (fired[src_neuron_id.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }
            input[id] += static_cast<activity_type>(synapse_strength);
        }

        for (const auto& [src_neuron_id, synapse_strength] : local_in_edges_plastic) {
            if (fired[src_neuron_id.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }
            const auto scaled = static_cast<activity_type>(synapse_strength) * scale;
            input[id] += static_cast<activity_type>(scaled);
        }
    }
}

void SynapticScalingActivityInputCPU::update_distant_input(std::span<const FiredStatus> /*fired*/, std::span<activity_type> input, const NeuronID first, const NeuronID last) {
    if (mpiPP::MPIInfo::get_number_ranks() == 1) {
        return;
    }

    if (needs_scale_update) {
        ensure_scales();
    }

    const auto& network_graph = get_network_graph();

    for (const auto neuron_id : NeuronIDRange::range(first, last)) {
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

        const auto scale = scales[id];

        const auto local_input = (scale * static_cast<activity_type>(local_plastic_input)) + local_static_input;
        input[id] += local_input;
    }
}
