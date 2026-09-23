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

#include "neurons/input/ActivityInput.h"

#include <functional>
#include <memory>
#include <vector>

class ChoiceFunction;

class ActivityInputFactory {
public:
    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_combined_activity(std::vector<std::shared_ptr<ActivityInput>> inputs = {});

    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_constant_activity(double activity = 0.0);

    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_normal_activity(double mean = 0.0, double stddev = 1.0);

    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_fast_normal_activity(double mean = 0.0, double stddev = 1.0, std::size_t multiplier = 10);

    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_flexible_activity(std::vector<std::shared_ptr<ActivityInput>> inputs, std::unique_ptr<ChoiceFunction> choice_function);

    [[nodiscard]] static std::shared_ptr<ActivityInput> construct_scale_activity(std::function<double(double)> func, std::shared_ptr<ActivityInput> input = construct_constant_activity());
};
