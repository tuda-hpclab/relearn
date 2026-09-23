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

#include "neurons/input/ActivityInputBase.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/Timers.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <cstddef>
#include <memory>
#include <vector>

/**
 * GPU implementation of ActivityInput: update_input_range queues CUDA work on a stream and
 * returns the events it queued.
 */
class ActivityInputGPU : public ActivityInputBase<LazySyncedArray> {
public:
    using Base = ActivityInputBase<LazySyncedArray>;

    explicit ActivityInputGPU(const int _number_ranks)
        : Base(_number_ranks) { }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @exception Might throw a RelearnException
     */
    void update_input(step_type step, const std::shared_ptr<StreamWrapper>& stream) {
        update_input_range(step, NeuronID{ 0 }, NeuronID{ get_number_neurons() }, stream);
    }

    void update_input(step_type step) {
        auto def_stream = std::make_shared<StreamWrapper>();
        update_input(step, def_stream);
        cudaDeviceSynchronize_bridge();
        cuda_resolve_gpu_timers();
    }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Might throw a RelearnException
     */
    virtual std::vector<EventWrapper> update_input_range(step_type step, NeuronID first, NeuronID last, const std::shared_ptr<StreamWrapper>& stream) = 0;

    std::vector<EventWrapper> update_input_range(step_type step, NeuronID first, NeuronID last) {
        auto def_stream = std::make_shared<StreamWrapper>();
        auto events = update_input_range(step, first, last, def_stream);
        cudaDeviceSynchronize_bridge();
        cuda_resolve_gpu_timers();
        return events;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Base::record_memory_footprint(footprint);
        footprint->emplace("ActivityInput GPU", _input.get_memory_footprint());
    }

    virtual std::size_t get_offset() {
        return 0;
    }
};
