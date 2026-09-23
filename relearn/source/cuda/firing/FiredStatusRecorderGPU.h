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

#include "neurons/firing/FiredStatusRecorderBase.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/neuron_model/NeuronModels.h"
#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"

#include <memory>
#include <span>

namespace utility {
class MemoryFootprint;
}

class NeuronMonitor;

/**
 * GPU implementation of the fired-status recorder: the per-period fire counters are additionally
 * mirrored to the device (all_fired_recorders, laid out period-major), since GPU neuron models
 * increment them directly on the device instead of going through set_fired().
 */
class FiredStatusRecorderGPU : public FiredStatusRecorderBase<LazySyncedArray> {
public:
    using Base = FiredStatusRecorderBase<LazySyncedArray>;
    using Base::set_fired;

    ~FiredStatusRecorderGPU();

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

    /**
     * @brief Returns a read-only device pointer to the fired status. Does not mark the device copy modified.
     * @return The device pointer
     */
    [[nodiscard]] const FiredStatus* get_d_fired_const() const {
        return fired.get_device_ptr_const();
    }

    /**
     * @brief Returns a handle bundling the fired-status and per-period fire-recorder device pointers for a
     *      neuron model kernel to write through. Marks the fired status and the fire recorders as device-modified,
     *      since a kernel receiving this handle is expected to write to both.
     * @return The handle
     */
    [[nodiscard]] FiredRecorderHandle get_d_fired_handle() {
        auto* fired_ptr = fired.get_device_ptr();
        auto* fired_recorder_ptr = d_fired_recorder_ptrs.get_device_ptr();
        (void)all_fired_recorders.get_device_ptr();
        return FiredRecorderHandle{ fired_ptr, fired_recorder_ptr, static_cast<int>(number_fire_recorders) };
    }

private:
    LazySyncedArray<counter_type*> d_fired_recorder_ptrs;
    LazySyncedArray<counter_type> all_fired_recorders;
};
