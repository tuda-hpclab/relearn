/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronTypesAdapter.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/shuffle/shuffle.h"

#include "factory/random/random_factory.h"

#include <range/v3/range/conversion.hpp>

#include <cstddef>
#include <memory>
#include <random>
#include <span>

void NeuronTypesAdapter::disable_neurons(RelearnTypes::number_neurons_type number_neurons, std::shared_ptr<NeuronsExtraInfo> extra_infos, std::mt19937& mt) {
    const auto number_disabled = RandomFactory::get_random_integer<RelearnTypes::number_neurons_type>(0, number_neurons, mt);
    disable_neurons(number_neurons, number_disabled, extra_infos, mt);
}

void NeuronTypesAdapter::disable_neurons(RelearnTypes::number_neurons_type number_neurons, RelearnTypes::number_neurons_type number_disabled, const std::shared_ptr<NeuronsExtraInfo>& extra_infos, std::mt19937& mt) {
    const auto neuron_ids = NeuronIDRange::range(number_neurons) | ranges::to_vector | actions::shuffle(mt);
    extra_infos->set_disabled_neurons(std::span<const NeuronID>{ neuron_ids.data(), number_disabled });
}
