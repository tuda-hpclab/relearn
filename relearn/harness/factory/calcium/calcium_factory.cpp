/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "calcium_factory.h"

#include "neurons/calcium/AbsoluteDecayCalciumCalculator.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/calcium/RelativeDecayCalciumCalculator.h"
#include "util/NeuronID.h"

#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <cpp-utility/Interval.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <limits>
#include <memory>
#include <random>

RelearnTypes::calcium_type CalciumFactory::get_random_beta(std::mt19937& mt) {
    const auto beta_distr = boost::random::uniform_real_distribution<RelearnTypes::calcium_type>(CalciumCalculator::min_beta, CalciumCalculator::max_beta);
    return beta_distr(mt);
}

RelearnTypes::calcium_type CalciumFactory::get_random_tau_C(std::mt19937& mt) {
    const auto tau_C_distr = boost::random::uniform_real_distribution<RelearnTypes::calcium_type>(CalciumCalculator::min_tau_C, CalciumCalculator::max_tau_C);
    return tau_C_distr(mt);
}

unsigned int CalciumFactory::get_random_h(std::mt19937& mt) {
    const auto h_distr = boost::random::uniform_int_distribution<unsigned int>(CalciumCalculator::min_h, CalciumCalculator::max_h);
    return h_distr(mt);
}

std::unique_ptr<CalciumCalculator> CalciumFactory::construct_calcium_calculator_no_decay(const RelearnTypes::calcium_type initial, const RelearnTypes::calcium_type target) {
    auto calc = std::make_unique<CalciumCalculator>();
    calc->set_initial_calcium_calculator([initial](mpiPP::MPIRank, NeuronID::value_type) { return initial; });
    calc->set_target_calcium_calculator([target](mpiPP::MPIRank, NeuronID::value_type) { return target; });
    return calc;
}

std::unique_ptr<CalciumCalculator> CalciumFactory::construct_calcium_calculator_relative_decay(const RelearnTypes::calcium_type initial, const RelearnTypes::calcium_type target, const RelearnTypes::calcium_type decay_amount, const CalciumCalculator::step_type decay_step) {
    auto decay_interval = utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<utility::Interval<RelearnTypes::step_type>::step_type>::max(), .frequency = decay_step };
    auto calc = std::make_unique<RelativeDecayCalciumCalculator>(decay_amount, decay_interval);
    calc->set_initial_calcium_calculator([initial](mpiPP::MPIRank, NeuronID::value_type) { return initial; });
    calc->set_target_calcium_calculator([target](mpiPP::MPIRank, NeuronID::value_type) { return target; });
    return calc;
}

std::unique_ptr<CalciumCalculator> CalciumFactory::construct_calcium_calculator_absolute_decay(const RelearnTypes::calcium_type initial, const RelearnTypes::calcium_type target, const RelearnTypes::calcium_type decay_amount, const CalciumCalculator::step_type decay_step) {
    auto decay_interval = utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<utility::Interval<RelearnTypes::step_type>::step_type>::max(), .frequency = decay_step };
    auto calc = std::make_unique<AbsoluteDecayCalciumCalculator>(decay_amount, decay_interval);
    calc->set_initial_calcium_calculator([initial](mpiPP::MPIRank, NeuronID::value_type) { return initial; });
    calc->set_target_calcium_calculator([target](mpiPP::MPIRank, NeuronID::value_type) { return target; });
    return calc;
}
