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

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <memory>
#include <span>

class NetworkGraph;

class SynapticActivityInput : public ActivityInput {
    /**
     * Virtual class that acts as base for the different types of synaptic activity input calculators.
     * The child classes shall override the get_synaptic_input method to calculate the synaptic input for a single neuron based on their implementation.
     */
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new instance of type SynapticActivityInput with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    explicit SynapticActivityInput(std::shared_ptr<FiredStatusCommunicator> communicator)
        : fired_status_comm(std::move(communicator)) {
        RelearnException::check(fired_status_comm != nullptr, "SynapticInputCalculator::SynapticInputCalculator: communicator was empty.");
    }

    SynapticActivityInput(const SynapticActivityInput&) = default;
    SynapticActivityInput& operator=(const SynapticActivityInput&) = default;

    SynapticActivityInput(SynapticActivityInput&&) = default;
    SynapticActivityInput& operator=(SynapticActivityInput&&) = default;

    ~SynapticActivityInput() override = default;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its input
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) override {
        ActivityInput::set_extra_infos(new_extra_info);
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range(step_type step, NeuronID first, NeuronID last) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        ActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(ActivityInput);
        footprint->emplace("SynapticActivityInput", total_size);

        fired_status_comm->record_memory_footprint(footprint);
    }

protected:
    virtual void update_local_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) = 0;

    virtual void update_distant_input(std::span<const FiredStatus> fired, std::span<double> input, NeuronID first, NeuronID last) = 0;

    std::shared_ptr<FiredStatusCommunicator> fired_status_comm{};
};
