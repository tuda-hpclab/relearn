#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ActivityInput.h"

#include "neurons/helper/NeuronMonitor.h"
#include "types/StimulusTypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

/**
 * @brief Scales the input of another ActivityInput by a specified function
 *      Holds everything that is common to the CPU and GPU flavors; StimulationActivityInputCPU
 *      and StimulationActivityInputGPU add the parts that differ (update_input_range's signature
 *      and return type differ by build; the GPU flavor is currently unsupported at runtime).
 */
class StimulationActivityInputBase : public ActivityInput {
public:
    /**
     * @brief Constructs a new instance of type StimulationActivityInputBase with 0 neurons and the passed values for all parameters
     * @param stimulus_function The function that generates the stimuli
     */
    StimulationActivityInputBase(const int _number_ranks, RelearnTypes::stimuli_function_type stimulus_function)
        : ActivityInput(_number_ranks)
        , stimulus(std::move(stimulus_function)) {
    }

    StimulationActivityInputBase(const StimulationActivityInputBase&) = delete;
    StimulationActivityInputBase& operator=(const StimulationActivityInputBase&) = delete;

    StimulationActivityInputBase(StimulationActivityInputBase&&) = default;
    StimulationActivityInputBase& operator=(StimulationActivityInputBase&&) = default;

    ~StimulationActivityInputBase() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override {
        monitor.register_paramter("Stimulated Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
            const auto input = get_input_internal();
            return utility::cast<float>(input[neuron_id]); }, []() { }, []() { });
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

protected:
    RelearnTypes::stimuli_function_type stimulus;
};
