/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticEquallyWeightedActivityInputCPU.h"

#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <span>

void SynapticEquallyWeightedActivityInputCPU::update_local_input(std::span<const FiredStatus> fired,
                                                                 std::span<activity_type> input, const NeuronID first,
                                                                 const NeuronID last) {
    const auto& network_graph = get_network_graph();

    const auto first_id = first.get_neuron_id();
    const auto last_id = last.get_neuron_id();

#pragma omp parallel for shared(network_graph, fired, input, first_id, last_id) default(none)
    for (auto id = first_id; id < last_id; ++id) {
        activity_type inp{ 0 };
        const auto& [local_in_edges_plastic, local_in_edges_static] = network_graph->get_local_in_edges(id);

        for (const auto& [source_neuron_id, synapse_strength] : local_in_edges_static) {
            if (fired[source_neuron_id.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            inp += static_cast<activity_type>(synapse_strength);
        }

        for (const auto& [source_neuron_id, synapse_strength] : local_in_edges_plastic) {
            if (fired[source_neuron_id.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            inp += static_cast<RelearnTypes::activity_type>(synapse_strength);
        }

        input[id] = inp * synapse_conductance;
    }
}

void SynapticEquallyWeightedActivityInputCPU::update_distant_input(std::span<const FiredStatus> /*fired*/,
                                                                   std::span<activity_type> input, const NeuronID first,
                                                                   const NeuronID last) {
    if (mpiPP::MPIInfo::get_number_ranks() == 1) {
        return;
    }

    const auto& network_graph = get_network_graph();

    const auto first_id = first.get_neuron_id();
    const auto last_id = last.get_neuron_id();

#pragma omp parallel for shared(network_graph, input, first_id, last_id) default(none)
    for (auto id = first_id; id < last_id; ++id) { // NOLINT(openmp-exception-escape) - id ranges over a pre-validated, caller-controlled partition; RelearnException::check here guards a partitioning invariant, not user input
        auto distant_plastic_input = RelearnTypes::plastic_synapse_weight{ 0 };
        auto distant_static_input = RelearnTypes::static_synapse_weight{ 0 };

        const auto& [distant_in_edges_plastic, distant_in_edges_static] = network_graph->get_distant_in_edges(id);

        for (const auto& [key, synapse_strength] : distant_in_edges_static) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (!contains_id) {
                continue;
            }

            distant_static_input += synapse_strength;
        }

        for (const auto& [key, synapse_strength] : distant_in_edges_plastic) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (!contains_id) {
                continue;
            }

            distant_plastic_input += synapse_strength;
        }

        const auto distant_input = static_cast<activity_type>(distant_plastic_input) + distant_static_input;
        input[id] += distant_input * synapse_conductance;
    }
}
