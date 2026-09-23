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

#include "ActivityInputBase.h"

#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include <vector>

/**
 * CPU implementation of ActivityInput: update_input_range updates the input in place and returns
 * nothing.
 */
class ActivityInputCPU : public ActivityInputBase<std::vector> {
public:
    using Base = ActivityInputBase<std::vector>;

    explicit ActivityInputCPU(const int _number_ranks)
        : Base(_number_ranks) { }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @exception Might throw a RelearnException
     */
    void update_input(step_type step) {
        update_input_range(step, NeuronID{ 0 }, NeuronID{ get_number_neurons() });
    }

    /**
     * @brief Updates the input
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Might throw a RelearnException
     */
    virtual void update_input_range(step_type step, NeuronID first, NeuronID last) = 0;
};
