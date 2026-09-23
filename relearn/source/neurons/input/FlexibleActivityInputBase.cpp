/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FlexibleActivityInputBase.h"

#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/ActivityInput.h"

#include <cpp-utility/Cast.hpp>

void FlexibleActivityInputBase::register_neuron_monitor(NeuronMonitor& monitor) {
    for (const auto& potential_input : potential_inputs) {
        potential_input->register_neuron_monitor(monitor);
    }

    monitor.register_paramter("Flexible Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto input = get_input_internal();
        return utility::cast<float>(input[neuron_id]); }, []() { }, []() { });
}

void FlexibleActivityInputBase::create_neurons(const FlexibleActivityInputBase::number_neurons_type creation_count) {
    ActivityInput::create_neurons(creation_count);

    for (const auto& potential_input : potential_inputs) {
        potential_input->create_neurons(creation_count);
    }

    chooser->create_neurons(creation_count);
}
