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

#include "neurons/firing/FiredStatusCommunicator.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <memory>

class NeuronMonitor;

/**
 * Virtual class that acts as base for the different types of synaptic activity input calculators.
 * The child classes shall override the update_local_input/update_distant_input hooks to calculate
 * the synaptic input for a single neuron based on their implementation.
 *      Holds everything that is common to the CPU and GPU synaptic activity inputs;
 *      SynapticActivityInputCPU and SynapticActivityInputGPU add the parts that differ --
 *      update_local_input/update_distant_input don't just have different bodies by build, they
 *      have entirely different signatures and return types (CPU works on host spans, GPU works on
 *      device pointers and returns queued CUDA events), so they can't share one declaration here.
 */
class SynapticActivityInputBase : public ActivityInput {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    SynapticActivityInputBase(const SynapticActivityInputBase&) = delete;
    SynapticActivityInputBase& operator=(const SynapticActivityInputBase&) = delete;

    SynapticActivityInputBase(SynapticActivityInputBase&&) = default;
    SynapticActivityInputBase& operator=(SynapticActivityInputBase&&) = default;

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
    /**
     * @brief Constructs a new instance of type SynapticActivityInput with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @exception Throws a RelearnException if communicator is empty
     */
    explicit SynapticActivityInputBase(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator)
        : ActivityInput(_number_ranks)
        , fired_status_comm(std::move(communicator)) {
        RelearnException::check(fired_status_comm != nullptr, "SynapticInputCalculator::SynapticInputCalculator: communicator was empty.");
    }

    std::shared_ptr<FiredStatusCommunicator> fired_status_comm;
};
