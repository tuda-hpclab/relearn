#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/SynapticElementType.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"

/**
 * This struct represents a neuron loaded from a file. It is made up of:
 * (1) Its position
 * (2) Its id
 * (3) Its signal type
 */
struct LoadedNeuron {
    RelearnTypes::position_type pos{ 0 };
    NeuronID id{ NeuronID::uninitialized_id() };
    SignalType signal_type{ SignalType::Excitatory };
};

/**
 * This struct summarizes neurons loaded from a file. It is made up of:
 * The minimum and maximum of (x, y, z)-positions of the neurons,
 * the number of loaded excitatory neurons, and the number
 * of loaded inhibitory neurons
 */
struct LoadedNeuronsInfo {
    RelearnTypes::position_type minimum;
    RelearnTypes::position_type maximum;
    RelearnTypes::number_neurons_type number_excitatory_neurons{};
    RelearnTypes::number_neurons_type number_inhibitory_neurons{};
};
