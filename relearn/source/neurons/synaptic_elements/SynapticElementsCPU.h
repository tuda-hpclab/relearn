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

#include "SynapticElementsBase.h"

#include <vector>

/**
 * CPU implementation of SynapticElements: update_number_elements integrates the growth on the
 * host, and commit_updates() runs on the host, no stream involved.
 */
class SynapticElementsCPU : public SynapticElementsBase {
public:
    using SynapticElementsBase::SynapticElementsBase;

    /**
     * @brief Updates the number of elements for each neuron (the deltas); have to be commited via commit_updates(...).
     * @param current_step The current step
     * @param calcium The calcium levels for each neuron
     * @param target_calcium The target calcium levels for each neuron
     * @exception Throws a RelearnException if extra_infos is nullptr, or if calcium or target_calcium is not of the right size, or if disable_flags is not of the right size
     */
    void update_number_elements(step_type current_step, std::span<const calcium_type> calcium, std::span<const calcium_type> target_calcium);

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
            return axons->commit_updates();
        }

        const auto signal_type = get_signal_type(synaptic_element_type);
        return dendrites->commit_updates(signal_type);
    }
};
