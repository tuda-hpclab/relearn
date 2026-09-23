/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsAdapter.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/growthrate/ConstantGrowthrateCalculator.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <memory>
#include <random>
#include <span>

void SynapticElementsAdapter::increase_grown_axons(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::span<const RelearnTypes::grown_type> grown_elements) {
    RelearnException::check(synaptic_elements != nullptr, "SynapticElementsAdapter::increase_grown_axons: synaptic_elements is empty");
    RelearnException::check(synaptic_elements->get_size() == grown_elements.size(), "SynapticElementsAdapter::increase_grown_axons: grown_elements does not have the correct size");

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
        synaptic_elements->axons->axons_base.grown_elements[neuron_id] += grown_elements[neuron_id];
        synaptic_elements->axons->axons_base.vacant_elements[neuron_id] += static_cast<unsigned int>(grown_elements[neuron_id]);
    }
}

void SynapticElementsAdapter::increase_grown_dendrites(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::span<const RelearnTypes::grown_type> grown_elements, const SignalType signal_type) {
    RelearnException::check(synaptic_elements != nullptr, "SynapticElementsAdapter::increase_grown_dendrites: synaptic_elements is empty");
    RelearnException::check(synaptic_elements->get_size() == grown_elements.size(), "SynapticElementsAdapter::increase_grown_dendrites: grown_elements does not have the correct size");

    if (signal_type == SignalType::Excitatory) {
        for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
            synaptic_elements->dendrites->excitatory_dendrites.grown_elements[neuron_id] += grown_elements[neuron_id];
            synaptic_elements->dendrites->excitatory_dendrites.vacant_elements[neuron_id] += static_cast<unsigned int>(grown_elements[neuron_id]);
        }
    } else {
        for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
            synaptic_elements->dendrites->inhibitory_dendrites.grown_elements[neuron_id] += grown_elements[neuron_id];
            synaptic_elements->dendrites->inhibitory_dendrites.vacant_elements[neuron_id] += static_cast<unsigned int>(grown_elements[neuron_id]);
        }
    }
}

void SynapticElementsAdapter::increase_connected_axons(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::span<const unsigned int> connected_elements) {
    RelearnException::check(synaptic_elements != nullptr, "SynapticElementsAdapter::increase_connected_axons: synaptic_elements is empty");
    RelearnException::check(synaptic_elements->get_size() == connected_elements.size(), "SynapticElementsAdapter::increase_connected_axons: grown_elements does not have the correct size");

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
        synaptic_elements->axons->axons_base.grown_elements[neuron_id] += static_cast<RelearnTypes::grown_type>(connected_elements[neuron_id]);
        synaptic_elements->axons->axons_base.connected_elements[neuron_id] += connected_elements[neuron_id];
    }
}

void SynapticElementsAdapter::increase_connected_dendrites(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::span<const unsigned int> connected_elements, const SignalType signal_type) {
    RelearnException::check(synaptic_elements != nullptr, "SynapticElementsAdapter::increase_connected_dendrites: synaptic_elements is empty");
    RelearnException::check(synaptic_elements->get_size() == connected_elements.size(), "SynapticElementsAdapter::increase_connected_dendrites: grown_elements does not have the correct size");

    if (signal_type == SignalType::Excitatory) {
        for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
            synaptic_elements->dendrites->excitatory_dendrites.grown_elements[neuron_id] += static_cast<RelearnTypes::grown_type>(connected_elements[neuron_id]);
            synaptic_elements->dendrites->excitatory_dendrites.connected_elements[neuron_id] += connected_elements[neuron_id];
        }
    } else {
        for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < synaptic_elements->get_size(); neuron_id++) {
            synaptic_elements->dendrites->inhibitory_dendrites.grown_elements[neuron_id] += static_cast<RelearnTypes::grown_type>(connected_elements[neuron_id]);
            synaptic_elements->dendrites->inhibitory_dendrites.connected_elements[neuron_id] += connected_elements[neuron_id];
        }
    }
}

void SynapticElementsAdapter::grow_and_connect(const std::shared_ptr<SynapticElements>& synaptic_elements, std::mt19937& mt) {
    const auto number_neurons = synaptic_elements->get_size();

    const auto grown_axons = SynapticElementsFactory::get_grown_elements(number_neurons, mt);
    const auto connected_axons = SynapticElementsFactory::get_connected_elements(number_neurons, mt);

    const auto grown_excitatory_dendrites = SynapticElementsFactory::get_grown_elements(number_neurons, mt);
    const auto connected_excitatory_dendrites = SynapticElementsFactory::get_connected_elements(number_neurons, mt);

    const auto grown_inhibitory_dendrites = SynapticElementsFactory::get_grown_elements(number_neurons, mt);
    const auto connected_inhibitory_dendrites = SynapticElementsFactory::get_connected_elements(number_neurons, mt);

    SynapticElementsAdapter::increase_grown_axons(synaptic_elements, grown_axons);
    SynapticElementsAdapter::increase_grown_dendrites(synaptic_elements, grown_excitatory_dendrites, SignalType::Excitatory);
    SynapticElementsAdapter::increase_grown_dendrites(synaptic_elements, grown_inhibitory_dendrites, SignalType::Inhibitory);

    SynapticElementsAdapter::increase_connected_axons(synaptic_elements, connected_axons);
    SynapticElementsAdapter::increase_connected_dendrites(synaptic_elements, connected_excitatory_dendrites, SignalType::Excitatory);
    SynapticElementsAdapter::increase_connected_dendrites(synaptic_elements, connected_inhibitory_dendrites, SignalType::Inhibitory);
}

void SynapticElementsAdapter::set_growthrate_calculators(const std::shared_ptr<SynapticElements>& synaptic_elements) {
    auto growth_rate_calculator_axon = std::make_shared<ConstantGrowthrateCalculator>(SynapticElements::default_nu);
    auto growth_rate_calculator_dend_ex = std::make_shared<ConstantGrowthrateCalculator>(SynapticElements::default_nu);
    auto growth_rate_calculator_dend_in = std::make_shared<ConstantGrowthrateCalculator>(SynapticElements::default_nu);

    synaptic_elements->set_growthrate_calculator(growth_rate_calculator_axon, SynapticElementType::Axon);
    synaptic_elements->set_growthrate_calculator(growth_rate_calculator_dend_ex, SynapticElementType::DendriteExcitatory);
    synaptic_elements->set_growthrate_calculator(growth_rate_calculator_dend_in, SynapticElementType::DendriteInhibitory);
}
