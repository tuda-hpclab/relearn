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

#include "neurons/calcium/CalciumCalculator.h"

#include <cpp-utility/Cast.hpp>

#include <memory>
#include <random>

class CalciumFactory {
public:
    static RelearnTypes::calcium_type get_random_beta(std::mt19937& mt);

    static RelearnTypes::calcium_type get_random_tau_C(std::mt19937& mt);

    static unsigned int get_random_h(std::mt19937& mt);

    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_no_decay(RelearnTypes::calcium_type initial = utility::as<RelearnTypes::calcium_type>(0.4), RelearnTypes::calcium_type target = utility::as<RelearnTypes::calcium_type>(0.8));

    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_relative_decay(RelearnTypes::calcium_type initial = utility::as<RelearnTypes::calcium_type>(0.4), RelearnTypes::calcium_type target = utility::as<RelearnTypes::calcium_type>(0.8), RelearnTypes::calcium_type decay_amount = utility::as<RelearnTypes::calcium_type>(0.1), CalciumCalculator::step_type decay_step = 1);

    static std::unique_ptr<CalciumCalculator> construct_calcium_calculator_absolute_decay(RelearnTypes::calcium_type initial = utility::as<RelearnTypes::calcium_type>(0.4), RelearnTypes::calcium_type target = utility::as<RelearnTypes::calcium_type>(0.8), RelearnTypes::calcium_type decay_amount = utility::as<RelearnTypes::calcium_type>(0.1), CalciumCalculator::step_type decay_step = 1);
};
