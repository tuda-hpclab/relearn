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

#include "NormalActivityInputBase.h"

/**
 * CPU implementation of NormalActivityInput: values are drawn from the host RNG on demand.
 */
class NormalActivityInputCPU : public NormalActivityInputBase {
public:
    NormalActivityInputCPU(const int _number_ranks, const activity_type mean_input, const activity_type stddev_input)
        : NormalActivityInputBase(_number_ranks, mean_input, stddev_input) { }

    NormalActivityInputCPU(const NormalActivityInputCPU&) = delete;
    NormalActivityInputCPU& operator=(const NormalActivityInputCPU&) = delete;

    NormalActivityInputCPU(NormalActivityInputCPU&&) = default;
    NormalActivityInputCPU& operator=(NormalActivityInputCPU&&) = default;

    ~NormalActivityInputCPU() override = default;

    /**
     * @brief Updates the input for the neurons (disabled ones get 0.0, others the base input)
     * @param step The current update step
     * @param first The first neuron (including) that shall be updated
     * @param last The last neuron (excluding) that shall be updated
     * @exception Throws a RelearnException if the size of the NeuronExtraInfos do not match this' size
     */
    void update_input_range([[maybe_unused]] step_type step, NeuronID first, NeuronID last) override;
};
