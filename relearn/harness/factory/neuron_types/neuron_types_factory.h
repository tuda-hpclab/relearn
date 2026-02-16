#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/DistantNeuronRequests.h"

#include <random>
#include <vector>

class NeuronTypesFactory {
public:
    static ElementType get_random_element_type(std::mt19937& mt) noexcept;

    static SignalType get_random_signal_type(std::mt19937& mt) noexcept;

    static DistantNeuronRequest::TargetNeuronType get_random_target_neuron_type(std::mt19937& mt);

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, std::mt19937& mt);

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, size_t number_inactive, std::mt19937& mt);

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, std::mt19937& mt);

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, size_t number_disabled, std::mt19937& mt);
};
