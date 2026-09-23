/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NormalActivityInputBase.h"

#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"

#include <cpp-utility/Cast.hpp>

void NormalActivityInputBase::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Normal Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto input = get_input_internal();
        return utility::cast<float>(input[neuron_id]); }, []() { }, []() { });
}
