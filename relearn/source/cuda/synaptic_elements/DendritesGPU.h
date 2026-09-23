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

#include "neurons/synaptic_elements/DendritesBase.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>
#include <vector>

/**
 * GPU implementation of Dendrites: exposes a CUDA handle onto the underlying arrays, and
 * commit_updates() takes the stream to run on.
 */
class DendritesGPU : public DendritesBase {
public:
    SynapticElementsBaseCudaHandleConst get_cuda_handle_const(const SignalType signal_type) const {
        if (signal_type == SignalType::Excitatory) {
            return excitatory_dendrites.get_cuda_handle_const();
        }
        return inhibitory_dendrites.get_cuda_handle_const();
    }

    SynapticElementsBaseCudaHandle get_cuda_handle(const SignalType signal_type) {
        if (signal_type == SignalType::Excitatory) {
            return excitatory_dendrites.get_cuda_handle();
        }
        return inhibitory_dendrites.get_cuda_handle();
    }

    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates(const SignalType signal_type, const std::shared_ptr<StreamWrapper>& stream) {
        if (signal_type == SignalType::Excitatory) {
            return excitatory_dendrites.commit_updates(stream);
        }

        return inhibitory_dendrites.commit_updates(stream);
    }
};
