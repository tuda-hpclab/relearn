/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BoxBasedRandomSubdomainAssignment.h"

#include "neurons/enums/SynapticElementType.h"
#include "sim/LoadedNeuron.h"
#include "structure/Partition.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

std::pair<std::vector<LoadedNeuron>, BoxBasedRandomSubdomainAssignment::number_neurons_type>
BoxBasedRandomSubdomainAssignment::place_neurons_in_box(const box_size_type& offset, const box_size_type& length_of_box, const number_neurons_type number_neurons, NeuronID::value_type first_id) {
    const auto& [min, max] = partition->get_simulation_box_size();
    const auto& simulation_box_length_ = (max - min).get_maximum();

    const auto& [length_x, length_y, length_z] = length_of_box;

    RelearnException::check(length_x > 0.0, "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: length_of_box.x was not positive: {}", length_x);
    RelearnException::check(length_y > 0.0, "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: length_of_box.y was not positive: {}", length_y);
    RelearnException::check(length_z > 0.0, "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: length_of_box.z was not positive: {}", length_z);

    RelearnException::check(offset.check_in_box(min, max), "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: The offset was not within the box.");

    RelearnException::check(length_of_box.get_x() <= simulation_box_length_ && length_of_box.get_y() <= simulation_box_length_ && length_of_box.get_z() <= simulation_box_length_,
                            "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: Requesting to fill neurons where no simulation box is");

    const auto um_per_neuron = get_um_per_neuron();

    const auto box = length_of_box - offset;

    const auto neurons_on_x = static_cast<std::size_t>(round(box.get_x() / um_per_neuron));
    const auto neurons_on_y = static_cast<std::size_t>(round(box.get_y() / um_per_neuron));
    const auto neurons_on_z = static_cast<std::size_t>(round(box.get_z() / um_per_neuron));

    const auto calculated_num_neurons = neurons_on_x * neurons_on_y * neurons_on_z;
    RelearnException::check(calculated_num_neurons >= number_neurons, "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: Should emplace more neurons than space in box");

    constexpr auto max_short = std::numeric_limits<std::uint16_t>::max();
    RelearnException::check(neurons_on_x <= max_short && neurons_on_y <= max_short && neurons_on_z <= max_short, "BoxBasedRandomSubdomainAssignment::place_neurons_in_box: Should emplace more neurons in a dimension than possible");

    const auto desired_ex = get_requested_ratio_excitatory_neurons();

    const auto expected_number_in = number_neurons - static_cast<number_neurons_type>(std::ceil(static_cast<double>(number_neurons) * desired_ex));
    const auto expected_number_ex = number_neurons - expected_number_in;

    auto positions = std::vector<std::size_t>{};
    positions.reserve(calculated_num_neurons);

    for (auto x_it = std::size_t{ 0 }; x_it < neurons_on_x; x_it++) {
        for (auto y_it = std::size_t{ 0 }; y_it < neurons_on_y; y_it++) {
            for (auto z_it = std::size_t{ 0 }; z_it < neurons_on_z; z_it++) {
                auto random_position = std::size_t{ 0 };
                // NOLINTNEXTLINE
                random_position |= (z_it);
                // NOLINTNEXTLINE
                random_position |= (y_it << 16U);
                // NOLINTNEXTLINE
                random_position |= (x_it << 32U);
                positions.emplace_back(random_position);
            }
        }
    }

    RandomHolder::shuffle(RandomHolderKey::Subdomain, positions);

    const auto signal_types = ranges::views::concat(
                                  ranges::views::repeat_n(SignalType::Excitatory, static_cast<std::ptrdiff_t>(expected_number_ex)),
                                  ranges::views::repeat_n(SignalType::Inhibitory, static_cast<std::ptrdiff_t>(expected_number_in)))
                              | ranges::to_vector
                              | RandomHolder::shuffleAction(RandomHolderKey::Subdomain);

    const auto create_loaded_neuron =
        // NOT regular_invocable
        [&positions, max_short, um_per_neuron, &offset, &signal_types, first_id](const auto i) -> LoadedNeuron {
        const auto pos_bitmask = positions[i];
        const auto x_it = (pos_bitmask >> 32U) & max_short;
        const auto y_it = (pos_bitmask >> 16U) & max_short;
        const auto z_it = pos_bitmask & max_short;

        const auto x_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(x_it);
        const auto y_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(y_it);
        const auto z_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(z_it);

        auto pos_rnd = box_size_type{ x_pos_rnd, y_pos_rnd, z_pos_rnd };
        pos_rnd *= um_per_neuron;

        const auto pos = pos_rnd + offset;
        const auto signal_type = signal_types[i];

        return { pos, NeuronID{ false, i + first_id }, signal_type};
    };

    auto nodes = std::vector<LoadedNeuron>{};
    nodes.reserve(number_neurons);
    for (const auto& neuron_id : NeuronID::range_id(number_neurons)) {
        nodes.push_back(create_loaded_neuron(neuron_id));
    }

    return { std::move(nodes), expected_number_ex };
}
