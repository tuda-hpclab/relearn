/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsBase.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <span>

void SynapticElementsBase::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "SynapticElements::init: init() was called previously");
    RelearnException::check(number_neurons > 0, "SynapticElements::init: number_neurons must be > 0");

    size = number_neurons;

    axons->init(number_neurons);
    dendrites->init(number_neurons);
}

void SynapticElementsBase::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "SynapticElements::create_neurons: init() was not called before");
    RelearnException::check(creation_count > 0, "SynapticElements::create_neurons: creation_count must be > 0");

    const auto old_size = size;
    const auto new_size = old_size + creation_count;
    size = new_size;

    axons->create_neurons(creation_count);
    dendrites->create_neurons(creation_count);
}

void SynapticElementsBase::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    RelearnException::check(size > 0, "SynapticElements::disable_neurons: init() was not called before");
    if (disabled_neuron_ids.empty()) {
        return;
    }

    axons->disable_neurons(disabled_neuron_ids);
    dendrites->disable_neurons(disabled_neuron_ids);
}

void SynapticElementsBase::register_neuron_monitor(NeuronMonitor& monitor) {
    axons->register_neuron_monitor(monitor);
    dendrites->register_neuron_monitor(monitor);
}
