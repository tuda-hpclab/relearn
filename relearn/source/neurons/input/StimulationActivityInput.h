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

#include "ActivityInput.h"

#include "Types2.h"

#include "neurons/NeuronsExtraInfo.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <functional>
#include <memory>

/**
 * @brief Scales the input of another ActivityInput by a specified function
 */
class StimulationActivityInput : public ActivityInput {
public:
    /**
     * @brief Constructs a new instance of type StimulationActivityInput with 0 neurons and the passed values for all parameters
     * @param stimulus_function The function that generates the stimuli
     */
    StimulationActivityInput(RelearnTypes::stimuli_function_type stimulus_function)
        : stimulus(std::move(stimulus_function)) {
    }

    StimulationActivityInput(const StimulationActivityInput&) = default;
    StimulationActivityInput& operator=(const StimulationActivityInput&) = default;

    StimulationActivityInput(StimulationActivityInput&&) = default;
    StimulationActivityInput& operator=(StimulationActivityInput&&) = default;

    ~StimulationActivityInput() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] const step_type step, const NeuronID first, const NeuronID last) override {
        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_STIMULATION);
        const auto extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "StimulationActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        const auto input = get_input_internal();
        std::ranges::fill(std::next(input.begin(), static_cast<std::int64_t>(first.get_neuron_id())), std::next(input.begin(), static_cast<std::int64_t>(last.get_neuron_id())), 0.0);

        const auto& stim = stimulus(step);
        for (const auto& [targets, value] : stim) {
            for (const auto& target : targets) {
                const auto neuron_id = target.get_neuron_id();
                if (target >= first && target < last && disable_flags[neuron_id] != UpdateStatus::Disabled) {
                    input[neuron_id] += value;
                }
            }
        }

        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_STIMULATION);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("StimulationActivityInput", total_size);
    }

private:
    RelearnTypes::stimuli_function_type stimulus{};
};
