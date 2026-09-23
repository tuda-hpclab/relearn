/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "synaptic_elements_factory.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/growthrate/ConstantGrowthrateCalculator.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "util/NeuronID.h"

#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/random/random_factory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

RelearnTypes::grown_type SynapticElementsFactory::get_random_synaptic_element_count(std::mt19937& mt) {
    return RandomFactory::get_random_double(min_grown_elements, std::nextafter(max_grown_elements, max_grown_elements * RelearnTypes::grown_type{ 2 }), mt);
}

unsigned int SynapticElementsFactory::get_random_synaptic_element_connected_count(std::mt19937& mt) {
    return RandomFactory::get_random_integer<unsigned int>(min_connected_elements, max_connected_elements, mt);
}

std::vector<SignalType> SynapticElementsFactory::get_excitatory_signal_types(const RelearnTypes::number_neurons_type number_neurons) {
    auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    return signal_types;
}

std::vector<SignalType> SynapticElementsFactory::get_inhibitory_signal_types(const RelearnTypes::number_neurons_type number_neurons) {
    auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Inhibitory);
    return signal_types;
}

std::vector<SignalType> SynapticElementsFactory::get_signal_types(const RelearnTypes::number_neurons_type number_excitatory_neurons, const RelearnTypes::number_neurons_type number_inhibitory_neurons, std::mt19937& mt) {
    auto signal_types = std::vector<SignalType>(number_excitatory_neurons + number_inhibitory_neurons);

    std::fill(signal_types.begin(), signal_types.begin() + static_cast<std::ptrdiff_t>(number_excitatory_neurons), SignalType::Excitatory);
    std::fill(signal_types.begin() + static_cast<std::ptrdiff_t>(number_excitatory_neurons), signal_types.end(), SignalType::Inhibitory);

    RandomFactory::shuffle(signal_types, mt);

    return signal_types;
}

std::vector<RelearnTypes::grown_type> SynapticElementsFactory::get_grown_elements(RelearnTypes::number_neurons_type number_neurons, std::mt19937& mt) {
    auto grown_elements = std::vector<RelearnTypes::grown_type>(number_neurons);
    for (auto i = std::size_t{ 0 }; i < number_neurons; ++i) {
        grown_elements[i] = get_random_synaptic_element_count(mt);
    }

    return grown_elements;
}

std::vector<RelearnTypes::counter_type> SynapticElementsFactory::get_connected_elements(RelearnTypes::number_neurons_type number_neurons, std::mt19937& mt) {
    auto connected_elements = std::vector<RelearnTypes::counter_type>(number_neurons);
    for (auto i = std::size_t{ 0 }; i < number_neurons; ++i) {
        connected_elements[i] = get_random_synaptic_element_connected_count(mt);
    }

    return connected_elements;
}

std::shared_ptr<SynapticElements> SynapticElementsFactory::construct_synaptic_elements(std::vector<SignalType> signal_types) {
    auto axons = std::make_shared<Axons>();
    auto dendrites = std::make_shared<Dendrites>();

    auto synaptic_elements = std::make_shared<SynapticElements>(std::move(axons), std::move(dendrites));

    synaptic_elements->init(signal_types.size());
    synaptic_elements->set_signal_types(std::move(signal_types));

    return synaptic_elements;
}

std::shared_ptr<SynapticElements> SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(std::shared_ptr<NeuronsExtraInfo> extra_infos, std::vector<SignalType> signal_types, RelearnTypes::grown_type number_axons, RelearnTypes::grown_type number_dendrites) {
    auto axons = std::make_shared<Axons>();
    auto dendrites = std::make_shared<Dendrites>();

    auto synaptic_elements = std::make_shared<SynapticElements>(std::move(axons), std::move(dendrites));

    synaptic_elements->init(signal_types.size());
    synaptic_elements->set_signal_types(std::move(signal_types));

    for (auto i = std::size_t{ 0 }; i < synaptic_elements->get_signal_types().size(); ++i) { // for each neuron
        const auto& signal_type = synaptic_elements->get_signal_types()[i];

        const auto axon_type = get_synaptic_element_type(ElementType::Axon, signal_type);
        const auto dendrite_type = get_synaptic_element_type(ElementType::Dendrite, signal_type);

        synaptic_elements->add_to_delta(number_axons, i, axon_type);
        synaptic_elements->add_to_delta(number_dendrites, i, dendrite_type);
    }

    synaptic_elements->set_extra_infos(extra_infos);

    std::ignore = synaptic_elements->commit_updates(SynapticElementType::Axon);
    std::ignore = synaptic_elements->commit_updates(SynapticElementType::DendriteExcitatory);
    std::ignore = synaptic_elements->commit_updates(SynapticElementType::DendriteInhibitory);

    return synaptic_elements;
}
