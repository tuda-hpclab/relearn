/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticActivityInputBase.h"

#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"

#include <cpp-utility/Cast.hpp>

void SynapticActivityInputBase::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Synaptic Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
        // get_input(neuron_id), not _input[neuron_id] directly: the GPU build overrides get_input()
        // to combine d_input_local/d_input_distant (see SynapticActivityInputGPU.h) because _input
        // itself is never populated on that path -- reading _input here would read 0 forever.
        return utility::cast<float>(get_input(neuron_id)); }, []() { }, []() { });
}
