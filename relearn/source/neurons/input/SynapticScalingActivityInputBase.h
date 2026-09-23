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

#include "SynapticActivityInput.h"

#include "util/RelearnException.h"

#include <memory>
#include <span>
#include <vector>

/**
 * This class calculates the synaptic input with each synapse having a relative weight of the number
 * of incoming synapses to a neuron.
 * The synaptic input is the sum of all synapses whose source neuron fired in the last step.
 * Inhibitory synapses have a negative weight.
 *      Holds everything that is common to the CPU and GPU flavors; SynapticScalingActivityInputCPU
 *      and SynapticScalingActivityInputGPU add the parts that differ (update_local_input/
 *      update_distant_input's signature, get_scales()'s host-vs-device-mirror read, and the GPU
 *      flavor is currently unsupported at runtime -- it exists only so factories compile without
 *      needing their own #ifdef).
 *      Templated on the per-neuron array storage (Storage<T>): SynapticScalingActivityInputCPU
 *      instantiates it with std::vector, SynapticScalingActivityInputGPU with LazySyncedArray.
 */
template <template <typename...> class Storage>
class SynapticScalingActivityInputBase : public SynapticActivityInput {
public:
    /**
     * @brief Constructs a new instance of type SynapticInputCalculator with 0 neurons and the passed values for all parameters
     * @param communicator The communicator for the fired status of distant neurons, not nullptr
     * @param total_scale The total scale for all synapses, must be > 0
     * @exception Throws a RelearnException if communicator is empty
     */
    SynapticScalingActivityInputBase(const int _number_ranks, std::shared_ptr<FiredStatusCommunicator> communicator, const activity_type total_scale = 1.0)
        : SynapticActivityInput(_number_ranks, std::move(communicator))
        , t_scale{ total_scale } {
        RelearnException::check(total_scale > activity_type{ 0 }, "Total scale must be > 0");
    }

    SynapticScalingActivityInputBase(const SynapticScalingActivityInputBase&) = delete;
    SynapticScalingActivityInputBase& operator=(const SynapticScalingActivityInputBase&) = delete;

    SynapticScalingActivityInputBase(SynapticScalingActivityInputBase&&) = default;
    SynapticScalingActivityInputBase& operator=(SynapticScalingActivityInputBase&&) = default;

    ~SynapticScalingActivityInputBase() override = default;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or init(...) was called before
     */
    void init(const number_neurons_type number_neurons) override {
        SynapticActivityInput::init(number_neurons);

        scales.resize(number_neurons, 0.0);
        needs_scale_update = true;
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        SynapticActivityInput::create_neurons(creation_count);

        const auto old_size = scales.size();
        scales.resize(old_size + creation_count, 0.0);
        needs_scale_update = true;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        SynapticActivityInput::record_memory_footprint(footprint);

        const auto total_size = sizeof(*this) - sizeof(SynapticActivityInput) + scales.size() * sizeof(activity_type);
        footprint->emplace("SynapticScalingActivityInput", total_size);
    }

protected:
    // LazySyncedArray aliases to plain std::vector when CUDA is disabled. When enabled, it also
    // owns the device mirror; d_scales (inherited from SynapticActivityInputGPU) is refreshed from
    // scales.get_device_ptr() after every resize and after ensure_scales() recomputes it, instead
    // of a hand-rolled malloc_d_scales()/update_d_scales() pair.
    Storage<activity_type> scales;
    activity_type t_scale{ 1.0 };
    bool needs_scale_update{ false };
};
