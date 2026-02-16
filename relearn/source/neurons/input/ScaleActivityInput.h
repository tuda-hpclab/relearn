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
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <functional>
#include <memory>

/**
 * @brief Scales the input of another ActivityInput by a specified function
 */
class ScaleActivityInput : public ActivityInput {
public:
    /**
     * @brief Constructs a new instance of type ScaleActivityInput with 0 neurons and the passed values for all parameters
     * @param activity_input The activity input which should be scaled, not empty
     * @param scaling_function The function that scales the input, not empty
     * @exception Throws a RelearnException if any argument is empty
     */
    ScaleActivityInput(std::shared_ptr<ActivityInput> activity_input, std::function<double(double)> scaling_function)
        : other_input(std::move(activity_input))
        , scaler(std::move(scaling_function)) {
        RelearnException::check(other_input != nullptr, "SynapticInputCalculator::SynapticInputCalculator: activity_input was empty.");
        RelearnException::check(scaler != nullptr, "SynapticInputCalculator::SynapticInputCalculator: scaling_function was empty.");
    }

    ScaleActivityInput(const ScaleActivityInput&) = default;
    ScaleActivityInput& operator=(const ScaleActivityInput&) = default;

    ScaleActivityInput(ScaleActivityInput&&) = default;
    ScaleActivityInput& operator=(ScaleActivityInput&&) = default;

    ~ScaleActivityInput() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Initializes this instance to hold the given number of neurons.
     *      Also calls init() on the held ActivityInput
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        ActivityInput::init(number_neurons);
        other_input->init(number_neurons);
    }

    /**
     * @brief Additionally created the given number of neurons.
     *      Also calls create_neurons on the held ActivityInput
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        ActivityInput::create_neurons(creation_count);
        other_input->create_neurons(creation_count);
    }

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its input
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) override {
        ActivityInput::set_extra_infos(new_extra_info);
        other_input->set_extra_infos(new_extra_info);
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(const std::shared_ptr<NetworkGraph>& new_network_graph) override {
        ActivityInput::set_network_graph(new_network_graph);
        other_input->set_network_graph(new_network_graph);
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] const step_type step, const NeuronID first, const NeuronID last) override {
        other_input->update_input_range(step, first, last);

        Timers::start(TimerRegion::CALC_ACTIVITY_INPUT_SCALE);
        const auto extra_infos = get_extra_infos();
        const auto disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ScaleActivityInput::update_input_range: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        const auto _other_input = other_input->get_input();
        const auto input = get_input_internal();

        for (const auto neuron_id : NeuronID::range(first, last)) {
            input[neuron_id.get_neuron_id()] = scaler(_other_input[neuron_id.get_neuron_id()]);
        }
        Timers::stop_and_add(TimerRegion::CALC_ACTIVITY_INPUT_SCALE);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("ScaleActivityInput", total_size);

        other_input->record_memory_footprint(footprint);
    }

private:
    std::shared_ptr<ActivityInput> other_input{};
    std::function<double(double)> scaler{};
};
