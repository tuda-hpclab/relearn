/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
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

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_constant_activity(const double activity) {
    return std::make_shared<ConstantActivityInput>(activity);
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_normal_activity(const double mean, const double stddev) {
    return std::make_shared<NormalActivityInput>(mean, stddev);
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_fast_normal_activity(const double mean, const double stddev, const std::size_t multiplier) {
    return std::make_shared<FastNormalActivityInput>(mean, stddev, multiplier);
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_flexible_activity(std::vector<std::shared_ptr<ActivityInput>> inputs, std::unique_ptr<ChoiceFunction> choice_function) {
    return std::make_shared<FlexibleActivityInput>(inputs, std::move(choice_function));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_combined_activity(std::vector<std::shared_ptr<ActivityInput>> inputs) {
    return std::make_shared<CombinedActivityInput>(std::move(inputs));
}

std::shared_ptr<ActivityInput> ActivityInputFactory::construct_scale_activity(std::function<double(double)> func, std::shared_ptr<ActivityInput> input) {
    return std::make_shared<ScaleActivityInput>(std::move(input), std::move(func));
}
