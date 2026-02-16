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

#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

class SynapticElementsFactory {
public:
    constexpr static double min_grown_elements = 0.0;
    constexpr static double max_grown_elements = 10.0;

    constexpr static unsigned int min_connected_elements = 0U;
    constexpr static unsigned int max_connected_elements = 10U;

    static double get_random_synaptic_element_count(std::mt19937& mt);

    static unsigned int get_random_synaptic_element_connected_count(std::mt19937& mt);

    static std::vector<SignalType> get_excitatory_signal_types(std::size_t number_neurons);

    static std::vector<SignalType> get_inhibitory_signal_types(std::size_t number_neurons);

    static std::vector<SignalType> get_signal_types(std::size_t number_excitatory_neurons, std::size_t number_inhibitory_neurons, std::mt19937& mt);

    static std::vector<double> get_grown_elements(std::size_t number_neurons, std::mt19937& mt);

    static std::vector<unsigned int> get_connected_elements(std::size_t number_neurons, std::mt19937& mt);

    static std::shared_ptr<SynapticElements> construct_synaptic_elements(std::vector<SignalType> signal_types);

    static std::shared_ptr<SynapticElements> construct_synaptic_elements_with_fixed_number_axons_dendrites(std::vector<SignalType> signal_types, double number_axons, double number_dendrites);
};
