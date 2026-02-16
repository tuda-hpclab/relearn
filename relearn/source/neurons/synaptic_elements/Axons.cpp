/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Axons.h"

#include "Types.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/NeuronMonitor.h"

#include <cstddef>

void Axons::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "Axons::init: init() was called previously");
    RelearnException::check(number_neurons > 0, "Axons::init: number_neurons must be > 0");

    size = number_neurons;

    signal_types.resize(size);

    axons_base.init(number_neurons);
}

void Axons::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "Axons::create_neurons: init() must be called before create_neurons()");
    RelearnException::check(creation_count > 0, "Axons::create_neurons: creation_count must be > 0");

    const auto old_size = size;
    const auto new_size = old_size + creation_count;
    size = new_size;

    signal_types.resize(new_size);

    axons_base.create_neurons(creation_count);
}

void Axons::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    axons_base.disable_neurons(disabled_neuron_ids);
}

void Axons::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Axons (exc.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Excitatory) {
            return static_cast<float>(axons_base.get_grown_elements()[neuron_id]);
        }
        return 0.0F;
    });

    monitor.register_paramter("Axons (inh.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Inhibitory) {
            return static_cast<float>(axons_base.get_grown_elements()[neuron_id]);
        }
        return 0.0F;
    });

    monitor.register_paramter("Axons (exc.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Excitatory) {
            return static_cast<float>(axons_base.get_connected_elements()[neuron_id]);
        }
        return 0.0F;
    });

    monitor.register_paramter("Axons (inh.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) {
        if (const auto signal_type = signal_types[neuron_id]; signal_type == SignalType::Inhibitory) {
            return static_cast<float>(axons_base.get_connected_elements()[neuron_id]);
        }
        return 0.0F;
    });
}

RelearnTypes::position_type Axons::get_bouton_position(const number_neurons_type neuron_id, const std::size_t bouton_id) const {
    RelearnException::check(neuron_id < size, "Axons::get_bouton_position: neuron_id is too large");
    RelearnException::check(bouton_id == 0, "Axons::get_bouton_position: bouton_id must be 0");
    return extra_infos->get_position(NeuronID{ neuron_id });
}

RelearnTypes::position_type Axons::get_bouton_position(const number_neurons_type neuron_id) const {
    RelearnException::check(neuron_id < size, "Axons::get_bouton_position: neuron_id is too large");
    return extra_infos->get_position(NeuronID{ neuron_id });
}
