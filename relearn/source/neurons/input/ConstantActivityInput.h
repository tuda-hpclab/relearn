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

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/NeuronMonitor.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <vector>

/**
 * @brief Provides a constant activity input for the neurons
 */
class ConstantActivityInput : public ActivityInput {
public:
    using number_neurons_type = ActivityInput::number_neurons_type;
    using step_type = ActivityInput::step_type;

    static constexpr double default_constant_activity{ 0.0 };
    static constexpr double min_constant_activity{ -10000.0 };
    static constexpr double max_constant_activity{ 10000.0 };

    /**
     * @brief Constructs a new object with the given constant input
     * @param input The base input
     */
    explicit ConstantActivityInput(const double input)
        : ActivityInput()
        , base_input(input) { }

    ConstantActivityInput(const ConstantActivityInput&) = default;
    ConstantActivityInput& operator=(const ConstantActivityInput&) = default;

    ConstantActivityInput(ConstantActivityInput&&) = default;
    ConstantActivityInput& operator=(ConstantActivityInput&&) = default;

    ~ConstantActivityInput() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override {
        monitor.register_paramter("Constant input", [this](const RelearnTypes::number_neurons_type) { return static_cast<float>(base_input); });
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] step_type step, NeuronID first, NeuronID last) override {
        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_CONSTANT);

        const auto& extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ConstantActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        const auto input = get_input_internal();

        for (const auto neuron_id : NeuronID::range(first, last)) {
            input[neuron_id.get_neuron_id()] = disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled ? 0.0 : base_input;
        }

        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_CONSTANT);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("ConstantActivityInput", total_size);
    }

    /**
     * Returns the constant activity level
     * @return Constant activity level
     */
    [[nodiscard]] double get_constant() const noexcept {
        return base_input;
    }

private:
    double base_input{ default_constant_activity };
};
