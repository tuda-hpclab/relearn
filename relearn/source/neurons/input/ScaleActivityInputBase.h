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

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/NeuronMonitor.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

/**
 * @brief Scales the input of another ActivityInput by a specified function
 *      Holds everything that is common to the CPU and GPU flavors; ScaleActivityInputCPU and
 *      ScaleActivityInputGPU add the parts that differ: the CPU side scales via an arbitrary
 *      std::function, the GPU side dispatches a fixed CudaConfig::scaling_function_enum to a
 *      kernel (functors can't cross into device code), and update_input_range's signature and
 *      return type differ by build.
 */
class ScaleActivityInputBase : public ActivityInput {
public:
    using activity_type = ActivityInput::activity_type;

    ScaleActivityInputBase(const int _number_ranks, std::shared_ptr<ActivityInput> activity_input)
        : ActivityInput(_number_ranks)
        , other_input(std::move(activity_input)) {
        RelearnException::check(other_input != nullptr, "SynapticInputCalculator::SynapticInputCalculator: activity_input was empty.");
    }

    ScaleActivityInputBase(const ScaleActivityInputBase&) = delete;
    ScaleActivityInputBase& operator=(const ScaleActivityInputBase&) = delete;

    ScaleActivityInputBase(ScaleActivityInputBase&&) = default;
    ScaleActivityInputBase& operator=(ScaleActivityInputBase&&) = default;

    ~ScaleActivityInputBase() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override {
        other_input->register_neuron_monitor(monitor);

        monitor.register_paramter("Scaled Input", [this](const RelearnTypes::number_neurons_type neuron_id) {
            const auto input = get_input_internal();
            return utility::cast<float>(input[neuron_id]); }, []() { }, []() { });
    }

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
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("ScaleActivityInput", total_size);

        other_input->record_memory_footprint(footprint);
    }

protected:
    std::shared_ptr<ActivityInput> other_input{};
};
