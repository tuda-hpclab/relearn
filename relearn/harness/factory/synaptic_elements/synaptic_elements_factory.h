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

#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

class SynapticElementsFactory {
public:
    constexpr static RelearnTypes::grown_type min_grown_elements = 0.0;
    constexpr static RelearnTypes::grown_type max_grown_elements = 10.0;

    constexpr static unsigned int min_connected_elements = 0U;
    constexpr static unsigned int max_connected_elements = 10U;

    static RelearnTypes::grown_type get_random_synaptic_element_count(std::mt19937& mt);

    static unsigned int get_random_synaptic_element_connected_count(std::mt19937& mt);

    static std::vector<SignalType> get_excitatory_signal_types(RelearnTypes::number_neurons_type number_neurons);

    static std::vector<SignalType> get_inhibitory_signal_types(RelearnTypes::number_neurons_type number_neurons);

    static std::vector<SignalType> get_signal_types(RelearnTypes::number_neurons_type number_excitatory_neurons, RelearnTypes::number_neurons_type number_inhibitory_neurons, std::mt19937& mt);

    static std::vector<RelearnTypes::grown_type> get_grown_elements(RelearnTypes::number_neurons_type number_neurons, std::mt19937& mt);

    static std::vector<RelearnTypes::counter_type> get_connected_elements(RelearnTypes::number_neurons_type number_neurons, std::mt19937& mt);

    static std::shared_ptr<SynapticElements> construct_synaptic_elements(std::vector<SignalType> signal_types);

    static std::shared_ptr<SynapticElements> construct_synaptic_elements_with_fixed_number_axons_dendrites(std::shared_ptr<NeuronsExtraInfo> extra_infos, std::vector<SignalType> signal_types, RelearnTypes::grown_type number_axons, RelearnTypes::grown_type number_dendrites);
};
