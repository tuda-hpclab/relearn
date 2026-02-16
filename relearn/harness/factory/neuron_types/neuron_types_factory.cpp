/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neuron_types_factory.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "util/shuffle/shuffle.h"

#include "factory/random/random_factory.h"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <cstddef>
#include <random>
#include <vector>

ElementType NeuronTypesFactory::get_random_element_type(std::mt19937& mt) noexcept {
    return RandomFactory::get_random_bool(mt) ? ElementType::Axon : ElementType::Dendrite;
}

SignalType NeuronTypesFactory::get_random_signal_type(std::mt19937& mt) noexcept {
    return RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
}

DistantNeuronRequest::TargetNeuronType NeuronTypesFactory::get_random_target_neuron_type(std::mt19937& mt) {
    const auto drawn = RandomFactory::get_random_bool(mt);

    if (drawn) {
        return DistantNeuronRequest::TargetNeuronType::Leaf;
    }

    return DistantNeuronRequest::TargetNeuronType::VirtualNode;
}

std::vector<FiredStatus> NeuronTypesFactory::get_fired_status(size_t number_neurons, std::mt19937& mt) {
    const auto number_disabled = RandomFactory::get_random_integer<size_t>(0, number_neurons, mt);
    return get_fired_status(number_neurons, number_disabled, mt);
}

std::vector<FiredStatus> NeuronTypesFactory::get_fired_status(size_t number_neurons, size_t number_inactive, std::mt19937& mt) {
    const auto num_inactive = static_cast<std::ptrdiff_t>(number_inactive);
    const auto num_active = static_cast<std::ptrdiff_t>(number_neurons - number_inactive);

    return ranges::views::concat(
               ranges::views::repeat_n(FiredStatus::Inactive, num_inactive),
               ranges::views::repeat_n(FiredStatus::Fired, num_active))
           | ranges::to_vector | actions::shuffle(mt);
}

std::vector<UpdateStatus> NeuronTypesFactory::get_update_status(size_t number_neurons, std::mt19937& mt) {
    const auto number_disabled = RandomFactory::get_random_integer<size_t>(0, number_neurons, mt);
    return get_update_status(number_neurons, number_disabled, mt);
}

std::vector<UpdateStatus> NeuronTypesFactory::get_update_status(size_t number_neurons, size_t number_disabled, std::mt19937& mt) {
    const auto num_disabled = static_cast<std::ptrdiff_t>(number_disabled);
    const auto num_enabled = static_cast<std::ptrdiff_t>(number_neurons - number_disabled);

    return ranges::views::concat(
               ranges::views::repeat_n(UpdateStatus::Disabled, num_disabled),
               ranges::views::repeat_n(UpdateStatus::Enabled, num_enabled))
           | ranges::to_vector | actions::shuffle(mt);
}
