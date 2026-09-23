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

#include "neurons/input/SynapticEquallyWeightedActivityInputBase.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>
#include <optional>

/**
 * GPU implementation of SynapticEquallyWeightedActivityInput: update_local_input/
 * update_distant_input launch a kernel via launch(...).
 */
class SynapticEquallyWeightedActivityInputGPU : public SynapticEquallyWeightedActivityInputBase {
public:
    using SynapticEquallyWeightedActivityInputBase::SynapticEquallyWeightedActivityInputBase;

protected:
    std::optional<EventWrapper> update_local_input(number_neurons_type first, number_neurons_type last, activity_type* d_input,
                                                   const FiredStatus* d_fired, const activity_type* d_scales,
                                                   const std::shared_ptr<StreamWrapper>& stream_wrapper) override;

    std::optional<EventWrapper> update_distant_input(const number_neurons_type first, const number_neurons_type last, activity_type* d_input, const activity_type* /*d_scales*/,
                                                     std::unique_ptr<FireStatusCommunicatorHandle>& fire_status_handle, const std::shared_ptr<StreamWrapper>& stream_wrapper) override;
};
