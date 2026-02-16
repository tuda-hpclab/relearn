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

#include "util/NeuronID.h"

#include <random>
#include <unordered_set>
#include <vector>

class NeuronIdFactory {
public:
    constexpr static int upper_bound_num_neurons = 1000;

    static NeuronID::value_type get_random_number_neurons(std::mt19937& mt);

    static NeuronID get_random_neuron_id(std::mt19937& mt);

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, std::mt19937& mt);

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID::value_type offset, std::mt19937& mt);

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID except, std::mt19937& mt);

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, const std::unordered_set<NeuronID>& except, std::mt19937& mt);

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, const std::vector<NeuronID>& except, std::mt19937& mt);

    static std::unordered_set<NeuronID> get_random_neuron_ids(NeuronID::value_type number_neurons_in_sim, NeuronID::value_type number_neurons_in_sample, std::mt19937& mt);

    static std::unordered_set<NeuronID> get_random_neuron_ids(NeuronID::value_type number_neurons_in_sim, NeuronID::value_type number_neurons_in_sample, NeuronID except, std::mt19937& mt);
};
