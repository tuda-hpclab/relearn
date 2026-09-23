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

#include "neurons/synaptic_elements/AxonsBase.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>
#include <vector>

/**
 * GPU implementation of Axons: exposes a CUDA handle onto the underlying arrays, and
 * commit_updates() takes the stream to run on.
 */
class AxonsGPU : public AxonsBase<LazySyncedArray> {
public:
    using Base = AxonsBase<LazySyncedArray>;

    SynapticElementsBaseCudaHandleConst get_cuda_handle_const() const {
        return axons_base.get_cuda_handle_const();
    }

    SynapticElementsBaseCudaHandle get_cuda_handle() {
        return axons_base.get_cuda_handle();
    }

    [[nodiscard]] const SignalType* get_d_signal_types() const {
        return signal_types.get_device_ptr_const();
    }

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates(const std::shared_ptr<StreamWrapper>& stream) {
        return axons_base.commit_updates(stream);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Base::record_memory_footprint(footprint);
        footprint->emplace("Axons GPU", signal_types.get_memory_footprint());
    }
};
