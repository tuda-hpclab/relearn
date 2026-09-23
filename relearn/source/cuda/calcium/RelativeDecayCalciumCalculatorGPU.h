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

#include "neurons/calcium/RelativeDecayCalciumCalculatorBase.h"

/**
 * GPU implementation of RelativeDecayCalciumCalculator: update_target_calcium launches the decay
 * kernel on the device target-calcium array.
 */
class RelativeDecayCalciumCalculatorGPU : public RelativeDecayCalciumCalculatorBase {
public:
    using RelativeDecayCalciumCalculatorBase::RelativeDecayCalciumCalculatorBase;

protected:
    void update_target_calcium(step_type step) noexcept override;
};
