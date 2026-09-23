#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/models/NeuronModelBase.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>

/**
 * GPU implementation of NeuronModel: update_electrical_activity(_benchmark) drives ActivityInput
 * on a dedicated stream, and device pointers to x/input are exposed for the concrete models'
 * kernels.
 */
class NeuronModelGPU : public NeuronModelBase<LazySyncedArray> {
public:
    using Base = NeuronModelBase<LazySyncedArray>;

    NeuronModelGPU(unsigned int h,
                   std::shared_ptr<ActivityInput>&& activity_input,
                   std::shared_ptr<FiredStatusCommunicator>&& fired_status_communicator)
        : Base(h, std::move(activity_input), std::move(fired_status_communicator)) {
        act_input_stream = std::make_shared<StreamWrapper>();
    }

    /**
     * @brief Performs one step of simulating the electrical activity for all neurons.
     *      This method performs communication via MPI.
     * @param step The current update step
     */
    void update_electrical_activity(step_type step);

    /**
     * @brief This function is used for benchmarking purposes
     * @param step The current update step
     */
    void update_electrical_activity_benchmark(step_type step);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

protected:
    /**
     * @brief Returns a device pointer to the membrane potential, marking it device-modified since a
     *      neuron model kernel receiving it is expected to write the newly integrated value back.
     * @return The device pointer
     */
    [[nodiscard]] activity_type* get_d_x() {
        return x.get_device_ptr();
    }

    /**
     * @brief Returns a read-only device pointer to the membrane potential. Does not mark the device copy modified.
     * @return The device pointer
     */
    [[nodiscard]] const activity_type* get_d_x_const() const {
        return x.get_device_ptr_const();
    }

    /**
     * @brief Returns a device pointer to the accumulated synaptic input, marking it device-modified.
     *      Only models that clamp/consume the input in place (e.g. AEIF) need this non-const overload.
     * @return The device pointer
     */
    [[nodiscard]] activity_type* get_d_input() {
        return act_input->get_input_arr()[0]->get_device_ptr();
    }

    /**
     * @brief Returns a read-only device pointer to the accumulated synaptic input. Does not mark the device copy modified.
     * @return The device pointer
     */
    [[nodiscard]] const activity_type* get_d_input_const() const {
        return act_input->get_input_arr_const()[0]->get_device_ptr_const();
    }

private:
    std::shared_ptr<StreamWrapper> act_input_stream;
};
