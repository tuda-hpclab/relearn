#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"
#include "Types2.h"

#include "neurons/input/ActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"

#include <functional>
#include <memory>
#include <string>

class ChoiceFunction;
class FiredStatusCommunicator;
class NetworkGraph;

struct ActivityInputParseContext {
    using scaling_function_type = std::function<double(double)>;

    double background_base = ConstantActivityInput::default_constant_activity;
    double background_mean = NormalActivityInput::default_mean_activity;
    double background_stddev = NormalActivityInput::default_stddev_activity;

    std::shared_ptr<FiredStatusCommunicator> communicator;

    scaling_function_type linear;
    scaling_function_type logarithmic;
    scaling_function_type hyperbolic_tangent;

    std::function<RelearnTypes::stimuli_function_type()> load_stimulus;

    std::function<std::pair<std::vector<std::shared_ptr<ActivityInput>>, std::unique_ptr<ChoiceFunction>>()> flexible_background;
};

/**
 * @brief Parse an expression that creates an ActivityInput according to the DSL.
 *
 * @details The DSL is a simple functional language, where each activity type can be created
 *          using it's name (e.g.: `NormalActivityInput` -> `normal`) and the required arguments to construct it.
 *          The parsing context provides variables that are partially exposed as variable names in the DSL and the rest
 *          is used as hidden arguments to constructors.
 *          Values inside the parsing context come from the CLI arguments passed to relearn and are the defaults otherwise.
 *          The DSL has the following structure (quotes " are  for illustrative purposes only):
 *
 *          activity -> const(double constant)
 *          activity -> normal(double mean, double stddev)
 *          activity -> fastnormal(double mean, double stddev, std::size_t multiplier)
 *          activity -> combined(activity, activities ...)
 *          activity -> synaptic
 *          activity -> scale(activity, scaling_function fun)
 *          activity -> stimulated
 *
 *          scaling_function -> "linear" | "logarithmic" | "hyperbolic_tangent"
 *          double -> "background_base" | "background_mean" | "background_stddev" | literal
 *          std::size_t -> literal
 *
 * @param input expression to parse
 * @param Context parse context that provides predefined arguments or used variables to ActivityInput's constructors
 * @return std::shared_ptr<ActivityInput> parsed ActivityInput
 * @exception RelearnException if the parsing failed
 */
[[nodiscard]] std::shared_ptr<ActivityInput> parse_activity(const std::string& input, const ActivityInputParseContext& Context);
