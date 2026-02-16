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

#include "Types.h"

#include "neurons/helper/ChoiceFunction.h"
#include "util/NeuronID.h"

#include <boost/functional/hash.hpp>

#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * Class that fulfills the requirements of a choice function for the FlexibleActivityInput.
 * It saves the current active set of activity indices and extracts the steps in which the activity input must be switched.
 * This way, we check for changes only when required and avoid additional overhead.
 */
class CachedChoiceFunction : public ChoiceFunction {
public:
    /**
     * Constructor
     * @param _background_activities This vector contains the changes of input activities as tuples of
     *      (1) Begin of the input (included)
     *      (2) end of the input (excluded; use numeric_limits::max for endless)
     *      (3) index of the input in the input vector
     *      (4) list of affected neuron ids
     * @param number_neurons The number of neurons on this rank
     */
    CachedChoiceFunction(std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, std::size_t, std::vector<NeuronID>>>&& _background_activities,
                         RelearnTypes::number_neurons_type number_neurons);

    [[nodiscard]] const std::unordered_set<std::size_t>& get_inputs_for_neuron_id(RelearnTypes::step_type step, NeuronID neuron_id) override;

    [[nodiscard]] const std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>& get_neuron_id_ranges_for_input(RelearnTypes::step_type step, std::size_t input_index) override;

    void create_neurons(RelearnTypes::number_neurons_type creation_count) override;

private:
    void update(RelearnTypes::step_type step);

    [[nodiscard]] bool is_update_necessary(RelearnTypes::step_type cur_step) const;

    [[nodiscard]] RelearnTypes::number_neurons_type get_number_neurons() const;

    [[nodiscard]] std::size_t get_number_inputs() const;

    std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, std::size_t, std::vector<NeuronID>>> background_activities{};
    std::size_t cur_update_steps_index{ 0 };
    std::vector<RelearnTypes::step_type> update_steps{};
    std::vector<std::unordered_set<std::size_t>> cur_input_indices{};
    std::vector<std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>> input_to_neuron_id_to_ranges{};

    RelearnTypes::number_neurons_type number_neurons{};
    std::size_t number_inputs{};
};