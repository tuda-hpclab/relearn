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

#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/input/SynapticIndividuallyWeightedActivityInputBase.h"

#include <memory>
#include <optional>

/**
 * GPU implementation of SynapticIndividuallyWeightedActivityInput. update_local_input/
 * update_distant_input are currently no-ops on the GPU (the weight map isn't used here).
 */
class SynapticIndividuallyWeightedActivityInputGPU : public SynapticIndividuallyWeightedActivityInputBase {
public:
    using SynapticIndividuallyWeightedActivityInputBase::SynapticIndividuallyWeightedActivityInputBase;

protected:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
    std::optional<EventWrapper> update_local_input(const number_neurons_type /*first*/, const number_neurons_type /*last*/, activity_type* /*d_input*/, const FiredStatus* /*d_fired*/, const activity_type* /*_d_scales*/, const std::shared_ptr<StreamWrapper>& /*stream_wrapper*/) override {
        return {};
    }

    std::optional<EventWrapper> update_distant_input(const number_neurons_type /*first*/, const number_neurons_type /*last*/, activity_type* /*d_input*/, const activity_type* /*d_scales*/,
                                                     std::unique_ptr<FireStatusCommunicatorHandle>& /*fire_status_handle*/, const std::shared_ptr<StreamWrapper>& /*stream_wrapper*/) override {
        return {};
    }
#pragma GCC diagnostic pop
};
