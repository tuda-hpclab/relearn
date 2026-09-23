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

#include "NeuronModelBase.h"

#include <vector>

/**
 * CPU implementation of NeuronModel: update_electrical_activity(_benchmark) drives ActivityInput
 * on the host, no stream involved.
 */
class NeuronModelCPU : public NeuronModelBase<std::vector> {
public:
    using Base = NeuronModelBase<std::vector>;
    using Base::Base;

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
};
