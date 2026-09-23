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

#include "FiredStatusRecorderBase.h"

#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"

#include <memory>
#include <span>
#include <vector>

namespace utility {
class MemoryFootprint;
}

class NeuronMonitor;

/**
 * CPU implementation of the fired-status recorder: the per-period fire counters live only on the
 * host.
 */
class FiredStatusRecorderCPU : public FiredStatusRecorderBase<std::vector> {
public:
    using Base = FiredStatusRecorderBase<std::vector>;
    using Base::set_fired;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or if init(...) has been called before
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Sets the fired status of the given neuron. Has only an effect if new_value == FiredStatus::Fired
     * @param neuron_id The neuron id to set the fired status for
     * @param new_value The status of the neuron
     * @exception Throws a RelearnException if the neuron_id is out of bounds
     */
    void set_fired(NeuronID neuron_id, FiredStatus new_value);

    /**
     * @brief Returns the number of times a neuron has fired in the given period
     * @param fire_recorder_period The period to get the data for
     * @return A span to the data
     */
    [[nodiscard]] std::span<const counter_type>
    get_fired_recorder(FireRecorderPeriod fire_recorder_period) const noexcept;

    /**
     * @brief Resets the data for the given period
     * @param period The period to reset
     */
    void reset(FireRecorderPeriod period);

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);
};
