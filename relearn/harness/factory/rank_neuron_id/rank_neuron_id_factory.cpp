/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "rank_neuron_id_factory.h"

#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronIDRange.h"

#include "adapter/helper/RankNeuronIdAdapter.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/rank_neuron_id/rank_neuron_id_factory.h"

#include <cstddef>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>

RankNeuronId RankNeuronIdFactory::generate_random_rank_neuron_id(std::mt19937& mt) {
    const auto rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto neuron_id = NeuronIdFactory::get_random_neuron_id(mt);

    return { rank, neuron_id };
}

std::pair<RankNeuronId, std::string> RankNeuronIdFactory::generate_random_rank_neuron_id_description(std::mt19937& mt) {
    auto rank_neuron_id = RankNeuronIdFactory::generate_random_rank_neuron_id(mt);
    auto description = RankNeuronIdAdapter::codify_rank_neuron_id(rank_neuron_id);
    return { rank_neuron_id, std::move(description) };
}

std::unordered_set<RankNeuronId> RankNeuronIdFactory::get_random_rank_neuron_ids(RelearnTypes::number_neurons_type number_neurons_per_rank, int num_ranks, RelearnTypes::number_neurons_type number_neurons_in_sample, std::mt19937& mt) {
    auto set = std::unordered_set<RankNeuronId>{};
    for ([[maybe_unused]] const auto _ : NeuronIDRange::range_id(number_neurons_in_sample)) {
        auto rni = RankNeuronId{};
        do {
            const auto neuron_id = NeuronIdFactory::get_random_neuron_id(number_neurons_per_rank, mt);
            const auto rank = MPIRankFactory::get_random_mpi_rank(num_ranks, mt);
            rni = RankNeuronId(rank, neuron_id);
        } while (set.contains(rni));
        set.insert(rni);
    }
    return set;
}
