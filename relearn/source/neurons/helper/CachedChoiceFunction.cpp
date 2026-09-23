/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CachedChoiceFunction.h"

#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include <boost/functional/hash.hpp>

#include <cpp-utility/data/group.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

CachedChoiceFunction::CachedChoiceFunction(std::vector<BackgroundActivityEntry>&& _background_activities, const RelearnTypes::number_neurons_type _number_neurons)
    : background_activities(std::move(_background_activities))
    , number_neurons(_number_neurons) {
    cur_input_indices.resize(number_neurons);

    // Extract begin and end of the inputs to store the steps in which we have to update
    update_steps = background_activities | ranges::views::transform([](const auto& p) { return p.begin; }) | ranges::to_vector;
    const auto ends = background_activities | ranges::views::transform([](const auto& p) { return p.end; }) | ranges::to_vector;
    update_steps.insert(update_steps.end(), ends.begin(), ends.end());
    std::sort(update_steps.begin(), update_steps.end());

    const auto last_us = std::unique(update_steps.begin(), update_steps.end());
    update_steps.erase(last_us, update_steps.end());

    const auto all_indices = background_activities | ranges::views::transform([](const auto& p) { return p.input_index; }) | ranges::to_vector;
    const auto largest_input_index_it = std::max_element(all_indices.begin(), all_indices.end());
    if (largest_input_index_it != all_indices.end()) {
        number_inputs = *(largest_input_index_it) + 1U;
    }

    // Check if there are duplicates
    for (const auto step : update_steps) {
        auto indices = std::vector<std::vector<size_t>>{};
        indices.resize(number_neurons);

        for (const auto& [begin, end, index, neuron_ids] : background_activities) {
            if (begin <= step && step < end) {
                for (const auto& neuron_id : neuron_ids) {
                    indices[neuron_id.get_neuron_id()].push_back(index);
                }
            }
        }
        for (const auto& neuron_id : NeuronIDRange::range_id(number_neurons)) {
            auto& indices_vector = indices[neuron_id];
            if (!indices_vector.empty()) {
                std::sort(indices_vector.begin(), indices_vector.end());
                const auto last_unique_it = std::unique(indices_vector.begin(), indices_vector.end());
                if (last_unique_it != indices_vector.end()) {
                    RelearnException::fail("BackgroundActivityIO::load_background_activity: The input with index {} was assigned more than once to the same neuron {} in step {}",
                                           *last_unique_it, neuron_id, step);
                }
            }
        }
    }

    input_to_neuron_id_to_ranges.resize(number_inputs);
}

const std::unordered_set<size_t>& CachedChoiceFunction::get_inputs_for_neuron_id(RelearnTypes::step_type step, NeuronID neuron_id) {
    if (!update_steps.empty() && step >= update_steps[cur_update_steps_index]) {
        RelearnException::check(step == update_steps[cur_update_steps_index], "BackgroundActivityIO::load_background_activity: Choice function was called out of order");

        // We need an update in this step
        std::fill(cur_input_indices.begin(), cur_input_indices.end(), std::unordered_set<size_t>{});

        // Update indices for all neuron ids
        for (const auto& [begin, end, index, neuron_ids] : background_activities) {
            if (begin <= step && step < end) {
                for (const auto cur_neuron_id : neuron_ids) {
                    cur_input_indices[cur_neuron_id.get_neuron_id()].insert(index);
                }
            }
        }
        cur_update_steps_index++;
    }

    // Return the cached indices
    return cur_input_indices[neuron_id.get_neuron_id()];
}

void CachedChoiceFunction::update(RelearnTypes::step_type step) {
    RelearnException::check(step == update_steps[cur_update_steps_index], "BackgroundActivityIO::load_background_activity: Choice function was called out of order");

    // We need an update in this step
    std::fill(cur_input_indices.begin(), cur_input_indices.end(), std::unordered_set<size_t>{});

    auto input_to_neuron_ids = std::vector<std::vector<NeuronID>>{};
    input_to_neuron_ids.resize(input_to_neuron_id_to_ranges.size(), {});

    // Update indices for all neuron ids
    for (const auto& [begin, end, index, neuron_ids] : background_activities) {
        if (begin <= step && step < end) {
            input_to_neuron_ids[index].insert(input_to_neuron_ids[index].end(), neuron_ids.begin(), neuron_ids.end());
            for (const auto cur_neuron_id : neuron_ids) {
                cur_input_indices[cur_neuron_id.get_neuron_id()].insert(index);
            }
        }
    }

    for (auto input_index = 0U; input_index < input_to_neuron_ids.size(); input_index++) {
        input_to_neuron_id_to_ranges[input_index] = std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>{};
        const auto& neuron_id_ranges = utility::group_consecutive_elements<NeuronID>(input_to_neuron_ids[input_index], [](const auto& predecessor, const auto& successor) { return predecessor.get_neuron_id() + 1 == successor.get_neuron_id() || predecessor == successor; });

        for (const auto& consecutive_neuron_ids : neuron_id_ranges) {
            input_to_neuron_id_to_ranges[input_index].emplace(consecutive_neuron_ids[0], consecutive_neuron_ids[consecutive_neuron_ids.size() - 1]);
        }
    }

    cur_update_steps_index++;
}

bool CachedChoiceFunction::is_update_necessary(RelearnTypes::step_type cur_step) const {
    return !update_steps.empty() && cur_step >= update_steps[cur_update_steps_index];
}

const std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>& CachedChoiceFunction::get_neuron_id_ranges_for_input(const RelearnTypes::step_type step, const size_t input_index) {
    if (is_update_necessary(step)) {
        update(step);
    }

    RelearnException::check(input_index < input_to_neuron_id_to_ranges.size(), "RelearnException::get_neuron_id_ranges_for_input: input_index is out of bounds {} / {}", input_index, input_to_neuron_id_to_ranges.size());

    return input_to_neuron_id_to_ranges[input_index];
}

RelearnTypes::number_neurons_type CachedChoiceFunction::get_number_neurons() const {
    return number_neurons;
}

size_t CachedChoiceFunction::get_number_inputs() const {
    return number_inputs;
}

void CachedChoiceFunction::create_neurons(RelearnTypes::number_neurons_type creation_count) {
    number_neurons = number_neurons + creation_count;
    cur_input_indices.resize(number_neurons);
}
