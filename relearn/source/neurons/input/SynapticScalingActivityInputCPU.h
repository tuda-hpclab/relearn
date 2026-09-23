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

#include "SynapticScalingActivityInputBase.h"

#include <span>
#include <vector>

/**
 * CPU implementation of SynapticScalingActivityInput: update_local_input/update_distant_input work
 * on host spans, recomputing the scales on demand.
 */
class SynapticScalingActivityInputCPU : public SynapticScalingActivityInputBase<std::vector> {
public:
    using Base = SynapticScalingActivityInputBase<std::vector>;
    using Base::Base;

    /**
     * @brief Returns a view of the scales
     * @return The scales
     */
    [[nodiscard]] std::span<const activity_type> get_scales() const noexcept {
        return scales;
    }

protected:
    void update_local_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;

    void update_distant_input(std::span<const FiredStatus> fired, std::span<activity_type> input, NeuronID first, NeuronID last) override;

private:
    void ensure_scales();
};
