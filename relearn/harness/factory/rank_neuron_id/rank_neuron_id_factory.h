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

#include "neurons/helper/RankNeuronId.h"
#include "types/BasicTypes.h"

#include <random>
#include <string>
#include <unordered_set>
#include <utility>

class RankNeuronIdFactory {
public:
    static RankNeuronId generate_random_rank_neuron_id(std::mt19937& mt);

    static std::pair<RankNeuronId, std::string> generate_random_rank_neuron_id_description(std::mt19937& mt);

    static std::unordered_set<RankNeuronId> get_random_rank_neuron_ids(RelearnTypes::number_neurons_type number_neurons_per_rank, int num_ranks,
                                                                       RelearnTypes::number_neurons_type number_neurons_in_sample, std::mt19937& mt);
};
