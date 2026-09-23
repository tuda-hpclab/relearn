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

#include "cuda/CudaConfig.h"
#include "io/BackgroundActivityIO.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "types/StimulusTypes.h"

#include <functional>
#include <memory>
#include <string>

class ChoiceFunction;
class FiredStatusCommunicator;
class NetworkGraph;

struct ActivityInputParseContext {
    using activity_type = RelearnTypes::activity_type;
    using scaling_function_type = std::function<activity_type(activity_type)>;

    activity_type background_base = ConstantActivityInput::default_constant_activity;
    activity_type background_mean = NormalActivityInput::default_mean_activity;
    activity_type background_stddev = NormalActivityInput::default_stddev_activity;

    std::shared_ptr<FiredStatusCommunicator> communicator;

    double synapse_conductance{};
#ifdef RELEARN_CUDA_ENABLED
    CudaConfig::scaling_function_enum linear;
    CudaConfig::scaling_function_enum logarithmic;
    CudaConfig::scaling_function_enum hyperbolic_tangent;
    double input_scale;
#else
    scaling_function_type linear;
    scaling_function_type logarithmic;
    scaling_function_type hyperbolic_tangent;
#endif

    std::function<RelearnTypes::stimuli_function_type()> load_stimulus{};

    std::function<LoadedBackgroundActivity()> flexible_background;
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
 *          activity -> const(activity_type constant)
 *          activity -> normal(activity_type mean, activity_type stddev)
 *          activity -> fastnormal(activity_type mean, activity_type stddev, std::size_t multiplier)
 *          activity -> combined(activity, activities ...)
 *          activity -> synaptic
 *          activity -> scale(activity, scaling_function fun)
 *          activity -> stimulated
 *
 *          scaling_function -> "linear" | "logarithmic" | "hyperbolic_tangent"
 *          activity_type -> "background_base" | "background_mean" | "background_stddev" | literal
 *          std::size_t -> literal
 *
 * @param input expression to parse
 * @param Context parse context that provides predefined arguments or used variables to ActivityInput's constructors
 * @return std::shared_ptr<ActivityInput> parsed ActivityInput
 * @exception RelearnException if the parsing failed
 */
[[nodiscard]] std::shared_ptr<ActivityInput> parse_activity(const std::string& input, const ActivityInputParseContext& Context);
