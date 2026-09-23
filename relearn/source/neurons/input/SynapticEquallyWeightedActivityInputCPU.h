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

#include "SynapticEquallyWeightedActivityInputBase.h"

#include <span>

/**
 * CPU implementation of SynapticEquallyWeightedActivityInput: update_local_input/
 * update_distant_input compute on host spans.
 */
class SynapticEquallyWeightedActivityInputCPU : public SynapticEquallyWeightedActivityInputBase {
public:
    using SynapticEquallyWeightedActivityInputBase::SynapticEquallyWeightedActivityInputBase;

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;
};
