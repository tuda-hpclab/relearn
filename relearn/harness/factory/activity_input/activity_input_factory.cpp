/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "activity_input_factory.h"

#include "neurons/input/ActivityInput.h"
#include "neurons/input/CombinedActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/FlexibleActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "neurons/input/ScaleActivityInput.h"

#include <cpp-utility/Cast.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

using activity_type = ActivityInput::activity_type;

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_constant_activity(const double activity) {
    return std::make_shared<ConstantActivityInput>(1, utility::cast<activity_type>(activity));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_normal_activity(const double mean, const double stddev) {
    return std::make_shared<NormalActivityInput>(1, utility::cast<activity_type>(mean), utility::cast<activity_type>(stddev));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_fast_normal_activity(const double mean, const double stddev, const std::size_t multiplier) {
    return std::make_shared<FastNormalActivityInput>(1, utility::cast<activity_type>(mean), utility::cast<activity_type>(stddev), multiplier);
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_flexible_activity(std::vector<std::shared_ptr<ActivityInput>> inputs, std::unique_ptr<ChoiceFunction> choice_function) {
    return std::make_shared<FlexibleActivityInput>(1, inputs, std::move(choice_function));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_combined_activity(std::vector<std::shared_ptr<ActivityInput>> inputs) {
    return std::make_shared<CombinedActivityInput>(1, std::move(inputs));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_scale_activity(std::function<double(double)> func, std::shared_ptr<ActivityInput> input) {
    // The scaling functions of the tests are written in double, the ones ScaleActivityInput takes
    // are written in the activity type, so the wrapper converts in both directions.
    auto scaler = [func = std::move(func)](const activity_type value) {
        return utility::cast<activity_type>(func(utility::cast<double>(value)));
    };

    return std::make_shared<ScaleActivityInput>(1, std::move(input), std::move(scaler));
}
