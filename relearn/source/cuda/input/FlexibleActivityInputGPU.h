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

#include "neurons/input/FlexibleActivityInputBase.h"
#include "cuda/input/ActivityInput.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/EventWrapper.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>
#include <vector>

/**
 * GPU implementation of FlexibleActivityInput. Currently unsupported at runtime; this class
 * exists only so factories compile without needing their own #ifdef around it.
 */
class FlexibleActivityInputGPU : public FlexibleActivityInputBase {
public:
    using FlexibleActivityInputBase::FlexibleActivityInputBase;

    std::vector<EventWrapper> update_input_range([[maybe_unused]] step_type step, [[maybe_unused]] NeuronID first, [[maybe_unused]] NeuronID last, [[maybe_unused]] const std::shared_ptr<StreamWrapper>& stream) override {
        CPU_NOT_SUPPORTED
    }

    using ActivityInput::update_input_range;
};
