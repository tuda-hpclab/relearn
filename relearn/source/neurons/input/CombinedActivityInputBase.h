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
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>
#include <vector>

/**
 * @brief This class allows to combine multiple sources of input into one
 *      Holds everything that is common to the CPU and GPU combined activity inputs;
 *      CombinedActivityInputCPU and CombinedActivityInputGPU add the parts that differ
 *      (update_input_range, and the device-side bookkeeping the GPU side needs to launch the
 *      combining kernel over its sub-inputs' device pointers).
 */
class CombinedActivityInputBase : public ActivityInput {
public:
    using number_neurons_type = ActivityInput::number_neurons_type;
    using step_type = ActivityInput::step_type;

    /**
     * @brief Constructs an object with the given sub inputs.
     *      Sums the input values of the objects.
     * @param inputs The inputs to hold, the vector can be empty, the pointers within not
     * @exception Throws a RelearnException if one of the inputs is empty
     */
    explicit CombinedActivityInputBase(const int _number_ranks, std::vector<std::shared_ptr<ActivityInput>> inputs)
        : ActivityInput(_number_ranks)
        , sub_inputs(std::move(inputs)) {
        for (const auto& sub_input : sub_inputs) {
            RelearnException::check(sub_input != nullptr, "CombinedActivityInputBase::CombinedActivityInputBase: One of the inputs is empty");
        }
    }

    CombinedActivityInputBase(const CombinedActivityInputBase&) = delete;
    CombinedActivityInputBase& operator=(const CombinedActivityInputBase&) = delete;

    CombinedActivityInputBase(CombinedActivityInputBase&&) = default;
    CombinedActivityInputBase& operator=(CombinedActivityInputBase&&) = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override {
        for (const auto& sub_input : sub_inputs) {
            sub_input->register_neuron_monitor(monitor);
        }

        monitor.register_paramter("Combined input", [this](const RelearnTypes::number_neurons_type neuron_id) {
            const auto input = get_input_internal();
            return utility::cast<float>(input[neuron_id]); }, []() { }, []() { });
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons.
     *      Also calls init() on each held ActivityInput
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        ActivityInput::init(number_neurons);

        for (const auto& sub_input : sub_inputs) {
            sub_input->init(number_neurons);
        }
    }

    /**
     * @brief Additionally created the given number of neurons.
     *      Also calls create_neurons on all held instances
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        ActivityInput::create_neurons(creation_count);

        for (const auto& sub_input : sub_inputs) {
            sub_input->create_neurons(creation_count);
        }
    }

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its input
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) override {
        ActivityInput::set_extra_infos(new_extra_info);

        for (const auto& sub_input : sub_inputs) {
            sub_input->set_extra_infos(new_extra_info);
        }
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(const std::shared_ptr<NetworkGraph>& new_network_graph) override {
        ActivityInput::set_network_graph(new_network_graph);

        for (const auto& sub_input : sub_inputs) {
            sub_input->set_network_graph(new_network_graph);
        }
    }

    /**
     * @brief Adds a source of input to the others
     * @param new_input The new source of input, not empty
     * @exception Throws a RelearnException if new_input is empty of the size of this and new_input do not math
     */
    void add_input_source(std::shared_ptr<ActivityInput> new_input) {
        RelearnException::check(new_input != nullptr, "CombinedActivityInputBase::add_input_source: The new input is empty");

        const auto my_size = get_number_neurons();
        const auto other_size = new_input->get_number_neurons();

        RelearnException::check(my_size == other_size, "CombinedActivityInputBase::add_input_source: I have {} neurons but the new input has {} neurons", my_size, other_size);

        const auto& network_graph = get_network_graph();
        if (network_graph != nullptr) {
            new_input->set_network_graph(network_graph);
        }

        const auto& extra_infos = get_extra_infos();
        if (extra_infos != nullptr) {
            new_input->set_extra_infos(extra_infos);
        }

        sub_inputs.emplace_back(std::move(new_input));
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto size_vector = sub_inputs.size() * sizeof(std::shared_ptr<ActivityInput>);
        const auto size_this = sizeof(*this) - sizeof(ActivityInput);
        const auto total_size = size_vector + size_this;

        footprint->emplace("CombinedActivityInput", total_size);

        for (const auto& sub_input : sub_inputs) {
            sub_input->record_memory_footprint(footprint);
        }
    }

protected:
    std::vector<std::shared_ptr<ActivityInput>> sub_inputs{};
};
