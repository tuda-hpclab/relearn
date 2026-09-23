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

#include "neurons/synaptic_elements/SynapticElementsBase.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>
#include <utility>
#include <vector>

/**
 * GPU implementation of SynapticElements: update_number_elements integrates the growth on the
 * device from calcium device pointers, and commit_updates() dispatches to a per-element-type
 * CUDA stream so the three commits can run concurrently.
 */
class SynapticElementsGPU : public SynapticElementsBase {
public:
    SynapticElementsGPU(std::shared_ptr<Axons> _axons, std::shared_ptr<Dendrites> _dendrites)
        : SynapticElementsBase(std::move(_axons), std::move(_dendrites)) {
        axons_stream = std::make_shared<StreamWrapper>();
        den_exc_stream = std::make_shared<StreamWrapper>();
        den_inh_stream = std::make_shared<StreamWrapper>();
    }

    /**
     * @brief Updates the number of elements for each neuron (the deltas); have to be commited via commit_updates(...).
     * @param d_calcium The calcium levels for each neuron, as a device pointer
     * @param d_target_calcium The target calcium levels for each neuron, as a device pointer
     */
    void update_number_elements(const calcium_type* d_calcium, const calcium_type* d_target_calcium);

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @param synaptic_element_type The type of the elements for which to commit the updates
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates(const SynapticElementType synaptic_element_type) {
        const auto element_type = get_element_type(synaptic_element_type);

        if (element_type == ElementType::Axon) {
            return axons->commit_updates(axons_stream);
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        const auto& stream = signal_type == SignalType::Excitatory ? den_exc_stream : den_inh_stream;
        return dendrites->commit_updates(signal_type, stream);
    }

private:
    std::shared_ptr<StreamWrapper> axons_stream;
    std::shared_ptr<StreamWrapper> den_exc_stream;
    std::shared_ptr<StreamWrapper> den_inh_stream;
};
