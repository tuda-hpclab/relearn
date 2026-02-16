/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neuron_id_factory.h"

#include "util/NeuronID.h"

#include "factory/random/random_factory.h"

#include <range/v3/algorithm/contains.hpp>

#include <random>
#include <unordered_set>
#include <vector>

NeuronID::value_type NeuronIdFactory::get_random_number_neurons(std::mt19937& mt) {
    return RandomFactory::get_random_integer<NeuronID::value_type>(1, upper_bound_num_neurons, mt);
}

NeuronID NeuronIdFactory::get_random_neuron_id(std::mt19937& mt) {
    const auto value = RandomFactory::get_random_integer<NeuronID::value_type>(0, upper_bound_num_neurons - 1, mt);
    return NeuronID{ value };
}

NeuronID NeuronIdFactory::get_random_neuron_id(NeuronID::value_type number_neurons, std::mt19937& mt) {
    const auto value = RandomFactory::get_random_integer<NeuronID::value_type>(0, number_neurons - 1, mt);
    return NeuronID{ value };
}

NeuronID NeuronIdFactory::get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID::value_type offset, std::mt19937& mt) {
    const auto value = RandomFactory::get_random_integer<NeuronID::value_type>(offset, offset + number_neurons - 1, mt);
    return NeuronID{ value };
}

NeuronID NeuronIdFactory::get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID except, std::mt19937& mt) {
    auto neuron_id = get_random_neuron_id(number_neurons, mt);
    while (neuron_id == except) {
        neuron_id = get_random_neuron_id(number_neurons, mt);
    }
    return neuron_id;
}

NeuronID NeuronIdFactory::get_random_neuron_id(NeuronID::value_type number_neurons, const std::unordered_set<NeuronID>& except, std::mt19937& mt) {
    auto neuron_id = get_random_neuron_id(number_neurons, mt);
    while (ranges::contains(except, neuron_id)) {
        neuron_id = get_random_neuron_id(number_neurons, mt);
    }
    return neuron_id;
}

NeuronID NeuronIdFactory::get_random_neuron_id(NeuronID::value_type number_neurons, const std::vector<NeuronID>& except, std::mt19937& mt) {
    auto neuron_id = get_random_neuron_id(number_neurons, mt);
    while (ranges::contains(except, neuron_id)) {
        neuron_id = get_random_neuron_id(number_neurons, mt);
    }
    return neuron_id;
}

std::unordered_set<NeuronID> NeuronIdFactory::get_random_neuron_ids(NeuronID::value_type number_neurons_in_sim, NeuronID::value_type number_neurons_in_sample, std::mt19937& mt) {
    auto set = std::unordered_set<NeuronID>{};
    for (auto i = 0U; i < number_neurons_in_sample; i++) {
        auto neuron_id = NeuronID{};
        do {
            neuron_id = get_random_neuron_id(number_neurons_in_sim, mt);
        } while (set.contains(neuron_id));
        set.insert(neuron_id);
    }
    return set;
}

std::unordered_set<NeuronID> NeuronIdFactory::get_random_neuron_ids(NeuronID::value_type number_neurons_in_sim, NeuronID::value_type number_neurons_in_sample, NeuronID except, std::mt19937& mt) {
    auto set = std::unordered_set<NeuronID>{};
    for ([[maybe_unused]] const auto _ : NeuronID::range_id(number_neurons_in_sample)) {
        auto neuron_id = NeuronID{};
        do {
            neuron_id = get_random_neuron_id(number_neurons_in_sim, mt);
        } while (set.contains(neuron_id) || neuron_id == except);
        set.insert(neuron_id);
    }
    return set;
}
