#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ActivityInput.h"

#include "neurons/helper/NeuronMonitor.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

/**
 * @brief Provides a constant activity input for the neurons
 *      Holds everything that is common to the CPU and GPU flavors; ConstantActivityInputCPU and
 *      ConstantActivityInputGPU add the parts that differ (update_input_range's signature and
 *      return type differ by build: CPU updates in place, GPU returns the CUDA events it queued).
 */
class ConstantActivityInputBase : public ActivityInput {
public:
    using activity_type = ActivityInput::activity_type;
    using number_neurons_type = ActivityInput::number_neurons_type;
    using step_type = ActivityInput::step_type;

    static constexpr activity_type default_constant_activity{ 0.0 };
    static constexpr activity_type min_constant_activity{ -10000.0 };
    static constexpr activity_type max_constant_activity{ 10000.0 };

    /**
     * @brief Constructs a new object with the given constant input
     * @param input The base input
     */
    explicit ConstantActivityInputBase(const int _number_ranks, const activity_type input)
        : ActivityInput(_number_ranks)
        , base_input(input) { }

    ConstantActivityInputBase(const ConstantActivityInputBase&) = delete;
    ConstantActivityInputBase& operator=(const ConstantActivityInputBase&) = delete;

    ConstantActivityInputBase(ConstantActivityInputBase&&) = default;
    ConstantActivityInputBase& operator=(ConstantActivityInputBase&&) = default;

    ~ConstantActivityInputBase() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override {
        monitor.register_paramter("Constant input", [this](const RelearnTypes::number_neurons_type) { return utility::cast<float>(base_input); }, []() { }, []() { });
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
    [[nodiscard]] activity_type get_constant() const noexcept {
        return base_input;
    }

protected:
    activity_type base_input{ default_constant_activity };
};
