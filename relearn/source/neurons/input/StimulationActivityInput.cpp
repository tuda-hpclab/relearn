/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "StimulationActivityInput.h"

#include "neurons/helper/NeuronMonitor.h"

void StimulationActivityInput::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Stimulated Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto input = get_input_internal();
        return static_cast<float>(input[neuron_id]);
    });
}
