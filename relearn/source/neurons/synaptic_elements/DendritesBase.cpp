/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "DendritesBase.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>

#include <cstddef>
#include <span>

void DendritesBase::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "Dendrites::init: init() was called previously");
    RelearnException::check(number_neurons > 0, "Dendrites::init: number_neurons must be > 0");

    size = number_neurons;

    excitatory_dendrites.init(number_neurons);
    inhibitory_dendrites.init(number_neurons);
}

void DendritesBase::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "Dendrites::create_neurons: init() must be called before create_neurons()");
    RelearnException::check(creation_count > 0, "Dendrites::create_neurons: creation_count must be > 0");

    const auto old_size = size;
    const auto new_size = old_size + creation_count;
    size = new_size;

    excitatory_dendrites.create_neurons(creation_count);
    inhibitory_dendrites.create_neurons(creation_count);
}

void DendritesBase::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    excitatory_dendrites.disable_neurons(disabled_neuron_ids);
    inhibitory_dendrites.disable_neurons(disabled_neuron_ids);
}

void DendritesBase::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Dendrites (exc.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(excitatory_dendrites.get_grown_elements()[neuron_id]); }, []() { }, []() { });

    monitor.register_paramter("Dendrites (inh.) grown", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(inhibitory_dendrites.get_grown_elements()[neuron_id]); }, []() { }, []() { });

    monitor.register_paramter("Dendrites (exc.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) { return static_cast<float>(excitatory_dendrites.get_connected_elements()[neuron_id]); }, []() { }, []() { });

    monitor.register_paramter("Dendrites (inh.) connected", [this](const RelearnTypes::number_neurons_type neuron_id) { return static_cast<float>(inhibitory_dendrites.get_connected_elements()[neuron_id]); }, []() { }, []() { });
}

RelearnTypes::position_type DendritesBase::get_spine_position(const number_neurons_type neuron_id, [[maybe_unused]] const std::size_t spine_id, [[maybe_unused]] const SignalType signal_type) const {
    RelearnException::check(neuron_id < size, "Dendrites::get_spine_position: neuron_id is too large");
    return extra_infos->get_position(NeuronID{ neuron_id });
}
