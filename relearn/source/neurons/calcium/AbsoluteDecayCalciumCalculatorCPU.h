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

#include "AbsoluteDecayCalciumCalculatorBase.h"

/**
 * CPU implementation of AbsoluteDecayCalciumCalculator: update_target_calcium decays the target
 * calcium in place on the host.
 */
class AbsoluteDecayCalciumCalculatorCPU : public AbsoluteDecayCalciumCalculatorBase {
public:
    using AbsoluteDecayCalciumCalculatorBase::AbsoluteDecayCalciumCalculatorBase;

protected:
    void update_target_calcium(step_type step) noexcept override;
};
