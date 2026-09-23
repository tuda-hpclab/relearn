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

#include "AxonsBase.h"

#include <vector>

/**
 * CPU implementation of Axons: commit_updates() runs on the host, no stream involved.
 */
class AxonsCPU : public AxonsBase<std::vector> {
public:
    /**
     * @brief Commits the deltas to the number of grown, connected, and vacant elements.
     *      After the call:
     *      (1) The deltas are 0.0 for each neuron
     *      (2) vacant + connected <= grown for each neuron
     * @return Returns the number of deletions for each neuron
     */
    [[nodiscard]] std::vector<counter_type> commit_updates() {
        return axons_base.commit_updates();
    }
};
