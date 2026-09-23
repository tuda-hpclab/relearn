/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "connector_factory.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "types/CommunicationTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"

#include <range/v3/view/indices.hpp>

#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

std::tuple<RelearnTypes::comm_map_creation<SynapseCreationRequest>, std::vector<size_t>, std::vector<size_t>> ConnectorFactory::create_incoming_requests(int number_ranks,
                                                                                                                                                 int current_rank, RelearnTypes::number_neurons_type number_neurons, size_t number_requests_lower_bound, size_t number_requests_upper_bound, std::mt19937& mt) {

    auto cm = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks);
    auto number_excitatory_requests = std::vector<size_t>(number_neurons, 0);
    auto number_inhibitory_requests = std::vector<size_t>(number_neurons, 0);

    for (const auto& target_id : NeuronIDRange::range(number_neurons)) {
        const auto number_requests = RandomFactory::get_random_integer<size_t>(number_requests_lower_bound, number_requests_upper_bound, mt);

        const auto id = target_id.get_neuron_id();

        for (const auto r : ranges::views::indices(number_requests)) {
            std::ignore = r; // This is here because GCC is complaining about an unused variable

            const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);
            const auto source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
            const auto fixed_source_id = (source_id.get_neuron_id() == target_id.get_neuron_id() && current_rank == source_rank.get_rank()) ? NeuronID{ (source_id.get_neuron_id() + 1) % number_neurons } : source_id;

            const auto signal_type = RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;

            const auto scr = SynapseCreationRequest{ target_id, fixed_source_id, signal_type };

            if (signal_type == SignalType::Excitatory) {
                number_excitatory_requests[id]++;
            } else {
                number_inhibitory_requests[id]++;
            }

            cm.append(source_rank, scr);
        }
    }

    return { cm, number_excitatory_requests, number_inhibitory_requests };
}
