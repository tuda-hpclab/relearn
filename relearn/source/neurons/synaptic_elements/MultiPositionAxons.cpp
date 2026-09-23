/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultiPositionAxons.h"

#include "neurons/synaptic_elements/Axons.h"
#include "types/SpaceTypes.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <cstddef>

void MultiPositionAxons::init(const number_neurons_type number_neurons) {
    Axons::init(number_neurons);

    if (number_axonal_boutons.size() <= number_neurons) {
        number_axonal_boutons.resize(number_neurons);
    }

    if (number_axonal_boutons_cumulative.size() <= number_neurons) {
        number_axonal_boutons_cumulative.resize(number_neurons);
    }
}

void MultiPositionAxons::create_neurons(const number_neurons_type creation_count) {
    Axons::create_neurons(creation_count);

    const auto old_size = get_size();
    const auto new_size = old_size + creation_count;

    if (number_axonal_boutons.size() <= new_size) {
        number_axonal_boutons.resize(creation_count);
    }

    if (number_axonal_boutons_cumulative.size() <= new_size) {
        number_axonal_boutons_cumulative.resize(creation_count);
    }
}

void MultiPositionAxons::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    Axons::disable_neurons(disabled_neuron_ids);

    // For now, it is not necessary to remove the bouton positions
}

MultiPositionAxons::counter_type MultiPositionAxons::get_number_boutons(const number_neurons_type neuron_id) const {
    const auto _size = get_size();
    RelearnException::check(neuron_id < _size, "MultiPositionAxons::get_number_boutons: neuron_id is too large: {}", neuron_id);

    return number_axonal_boutons[neuron_id];
}

RelearnTypes::position_type MultiPositionAxons::get_bouton_position(const number_neurons_type neuron_id, const std::size_t bouton_id) const {
    const auto _size = get_size();
    RelearnException::check(neuron_id < _size, "MultiPositionAxons::get_bouton_position: neuron_id is too large: {}", neuron_id);

    const auto bouton_size = number_axonal_boutons[neuron_id];
    RelearnException::check(bouton_id < bouton_size, "MultiPositionAxons::get_bouton_position: bouton_id is too large: {}", bouton_id);

    const auto index = number_axonal_boutons_cumulative[neuron_id] + bouton_id;

    return bouton_positions_flat[index];
}

RelearnTypes::position_type MultiPositionAxons::get_bouton_position(const number_neurons_type neuron_id) const {
    const auto _size = get_size();
    RelearnException::check(neuron_id < _size, "MultiPositionAxons::get_bouton_position: neuron_id is too large: {}", neuron_id);

    const auto bouton_size = number_axonal_boutons[neuron_id];
    RelearnException::check(bouton_size > 0, "MultiPositionAxons::get_bouton_position: neuron_id has no boutons: {}", neuron_id);

    const auto random_index = RandomHolder::get_random_uniform_integer(RandomHolderKey::SynapticElements, counter_type{ 0 }, bouton_size - 1);
    const auto index = number_axonal_boutons_cumulative[neuron_id] + random_index;

    return bouton_positions_flat[index];
}
