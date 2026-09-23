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

#include "neurons/input/CombinedActivityInputBase.h"
#include "cuda/input/ActivityInput.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <cstddef>
#include <memory>
#include <vector>

/**
 * GPU implementation of CombinedActivityInput: launches a kernel that sums each sub-input's
 * device-side input into this input's own. create_neurons() is not supported -- CombinedActivityInput
 * is only ever used for the (fixed-size) background activity inputs on the GPU path.
 */
class CombinedActivityInputGPU : public CombinedActivityInputBase {
public:
    explicit CombinedActivityInputGPU(const int _number_ranks, std::vector<std::shared_ptr<ActivityInput>> inputs)
        : CombinedActivityInputBase(_number_ranks, std::move(inputs)) {
        for (auto i = std::size_t{ 0 }; i < sub_inputs.size(); i++) {
            streams.push_back(std::make_shared<StreamWrapper>());
        }
    }

    CombinedActivityInputGPU(const CombinedActivityInputGPU&) = delete;
    CombinedActivityInputGPU& operator=(const CombinedActivityInputGPU&) = delete;

    CombinedActivityInputGPU(CombinedActivityInputGPU&&) = default;
    CombinedActivityInputGPU& operator=(CombinedActivityInputGPU&&) = default;

    ~CombinedActivityInputGPU() override = default;

    void init(const number_neurons_type number_neurons) override {
        CombinedActivityInputBase::init(number_neurons);
        resize_sub_input_ptrs();
    }

    void create_neurons([[maybe_unused]] const number_neurons_type creation_count) override {
        CPU_NOT_SUPPORTED
    }

    /**
     * @brief Adds a source of input to the others
     * @param new_input The new source of input, not empty
     * @exception Throws a RelearnException if new_input is empty of the size of this and new_input do not math
     */
    void add_input_source(std::shared_ptr<ActivityInput> new_input) {
        CombinedActivityInputBase::add_input_source(std::move(new_input));
        streams.push_back(std::make_shared<StreamWrapper>());
        resize_sub_input_ptrs();
    }

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    std::vector<EventWrapper> update_input_range(step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) override {

        events.clear();
        auto sub_input_size = sub_input_ptrs.size();

        auto j = 0U;
        for (auto i = 0U; i < sub_inputs.size(); i++) {
            auto _events = sub_inputs[i]->update_input_range(step, first, last, streams[i]);
            const auto arr = sub_inputs[i]->get_input_arr();
            for (auto k = 0U; k < arr.size(); k++) {
                sub_input_ptrs[j] = arr[k]->get_device_ptr();
                j++;
            }
            for (auto&& e : _events) {
                events.emplace_back(std::move(e));
            }
        }
        // No Timers wrap here: this only enqueues async work on `stream` (no sync), so a host
        // Timers::start/stop around it would close long before the GPU kernel actually finishes and
        // could never correctly bound the CUDA-event-timed CUDA_UPDATE_COMBINED_ACTIVITY_KERNEL
        // nested inside update_combined_activity_input_range_entry. Only the outer CALC_ACTIVITY_INPUT
        // (in NeuronModel.cpp, where the device is actually synced and GPU timers resolved) can.
        for (const auto& event : events) {
            event.wait_for_event(stream);
        }

        release_synaptic_activity_set();

        update_combined_activity_input_range_entry(static_cast<CudaConfig::number_neurons_type>(first.get_neuron_id()), static_cast<CudaConfig::number_neurons_type>(last.get_neuron_id()), sub_input_size, _input.get_device_ptr(), sub_input_ptrs.get_device_ptr(), stream);
        return {};
    }

private:
    // Device array of each sub-input's device pointer; rebuilt (host side, via operator[]) every
    // update_input_range() call since sub-inputs' underlying device buffers can move between
    // steps, then lazily re-uploaded to the device the next time get_device_ptr() is read.
    LazySyncedArray<activity_type*> sub_input_ptrs;

    std::vector<std::shared_ptr<StreamWrapper>> streams;
    std::vector<EventWrapper> events;

    void resize_sub_input_ptrs() {
        auto sub_input_size = std::size_t{ 0 };
        for (auto i = 0U; i < sub_inputs.size(); i++) {
            sub_input_size += sub_inputs[i]->get_input_arr().size();
        }
        sub_input_ptrs.resize(sub_input_size);
    }
};
