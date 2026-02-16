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

#include <unordered_set>
#include <vector>

/**
 * Class that fulfills the requirements of a choice function for the FlexibleActivityInput.
 * It saves the current active set of activity indices and extracts the steps in which the activity input must be switched.
 * This way, we check for changes only when required and avoid additional overhead.
 */
class ChoiceFunction {
public:
    ChoiceFunction() = default;

    ChoiceFunction(const ChoiceFunction&) = default;
    ChoiceFunction& operator=(const ChoiceFunction&) = default;

    ChoiceFunction(ChoiceFunction&&) = default;
    ChoiceFunction& operator=(ChoiceFunction&&) = default;

    virtual ~ChoiceFunction() = default;

    /**
     * Returns the input indices for the FlexibleActivityInput for a certain neuron and step
     * Method calls must use the same or a higher value of step compared to the last method call
     * @param step The current step
     * @param neuron_id The neuron id
     * @return Set of currently active inputs for the neuron id in this step
     */
    [[nodiscard]] virtual const std::unordered_set<size_t>& get_inputs_for_neuron_id(RelearnTypes::step_type step, NeuronID neuron_id) = 0;

    /**
     * Returns consecutive neuron id ranges which receive input from a certain Activity input at a time step
     * Method calls must use the same or a higher value of step compared to the last method call
     * @param step The current step
     * @param input_index The index of the ActivityInput
     * @return Set of neuron ranges as pair of (1) first NeuronId (included) and (2) last (excluded) neuron id. The ranges have no overlap
     */
    [[nodiscard]] virtual const std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>& get_neuron_id_ranges_for_input(RelearnTypes::step_type step, size_t input_index) = 0;

    /**
     * Call this method when new neurons were created
     * @param creation_count The number of added neurons to this mpi rank
     */
    virtual void create_neurons([[maybe_unused]] RelearnTypes::number_neurons_type creation_count) { }
};