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

#include "neurons/helper/ChoiceFunction.h"
#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <vector>

/**
 * @brief This class allows to flexibly choose none, one or multiple input sources
 *      Holds everything that is common to the CPU and GPU flavors; FlexibleActivityInputCPU and
 *      FlexibleActivityInputGPU add the part that differs (update_input_range's signature and
 *      return type; the GPU flavor is currently unsupported at runtime).
 */
class FlexibleActivityInputBase : public ActivityInput {
public:
    using number_neurons_type = ActivityInput::number_neurons_type;
    using step_type = ActivityInput::step_type;

    /**
     * @brief Constructs an object with the given sub inputs.
     *      Chooses the input values from the objects.
     * @param inputs Vector of activity inputs that must be referenced by the choice_function
     * @param choice_function Unique_ptr to the choice function. It determines which ActivityInputs are active for which neuron in which step. It returns the indices of the ActivityInputs in inputs
     * @exception Throws a RelearnException if one of the inputs is empty
     */
    FlexibleActivityInputBase(const int _number_ranks, std::vector<std::shared_ptr<ActivityInput>> inputs, std::unique_ptr<ChoiceFunction> choice_function)
        : ActivityInput(_number_ranks)
        , potential_inputs(std::move(inputs))
        , chooser(std::move(choice_function)) {
        for (const auto& potential_input : potential_inputs) {
            RelearnException::check(potential_input != nullptr, "FlexibleActivityInput::FlexibleActivityInput: One of the inputs is empty");
        }
        RelearnException::check(chooser != nullptr, "FlexibleActivityInput::FlexibleActivityInput: Choice function is empty");
    }

    FlexibleActivityInputBase(const FlexibleActivityInputBase&) = delete;
    FlexibleActivityInputBase& operator=(const FlexibleActivityInputBase&) = delete;

    FlexibleActivityInputBase(FlexibleActivityInputBase&&) = default;
    FlexibleActivityInputBase& operator=(FlexibleActivityInputBase&&) = default;

    ~FlexibleActivityInputBase() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Initializes this instance to hold the given number of neurons.
     *      Also calls init() on each held ActivityInput
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        ActivityInput::init(number_neurons);

        for (const auto& potential_input : potential_inputs) {
            potential_input->init(number_neurons);
        }
    }

    /**
     * @brief Additionally created the given number of neurons.
     *      Also calls create_neurons on all held instances
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(number_neurons_type creation_count) override;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its input
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) override {
        ActivityInput::set_extra_infos(new_extra_info);

        for (const auto& potential_input : potential_inputs) {
            potential_input->set_extra_infos(new_extra_info);
        }
    }

    /**
     * @brief Adds a source of input to the others
     * @param new_input The new source of input, not empty
     * @exception Throws a RelearnException if new_input is empty of the size of this and new_input do not math
     */
    void add_input_source(std::shared_ptr<ActivityInput> new_input) {
        RelearnException::check(new_input != nullptr, "FlexibleActivityInput::add_input_source: The new input is empty");

        const auto my_size = get_number_neurons();
        const auto other_size = new_input->get_number_neurons();

        RelearnException::check(my_size == other_size, "FlexibleActivityInput::add_input_source: I have {} neurons but the new input has {} neurons", my_size, other_size);

        potential_inputs.emplace_back(std::move(new_input));
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto size_vector = potential_inputs.size() * sizeof(std::shared_ptr<ActivityInput>);
        const auto size_this = sizeof(*this) - sizeof(ActivityInput);
        const auto total_size = size_vector + size_this;

        footprint->emplace("FlexibleActivityInput", total_size);

        for (const auto& potential_input : potential_inputs) {
            potential_input->record_memory_footprint(footprint);
        }
    }

protected:
    std::vector<std::shared_ptr<ActivityInput>> potential_inputs{};
    std::unique_ptr<ChoiceFunction> chooser{};
};
