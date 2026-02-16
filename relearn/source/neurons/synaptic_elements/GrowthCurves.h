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

/**
 * @brief A gaussian curve that is compressed by growth-factor nu and intersects the x-axis at
 *      eta (left intersection) and epsilon (right intersection).
 *      It is positive on (eta, epsilon) and negative at (-inf, eta), (epsilon, inf)
 *      Its maximum is at m := (eta + epsilon) / 2, and it is symmetric wrt. m.
 *      It attracts (-inf, eta) to -inf and (eta, inf) to epsilon.
 *      See Butz and van Ooyen, 2013 PloS Comp Biol, Equation 4.
 * @param current The current value (of calcium in the neuron)
 * @param eta The left intersection with the x-axis
 * @param epsilon The right intersection with the x-axis
 */
inline double gaussian_growth_curve(const double current, const double eta, const double epsilon) noexcept {
    if (eta == epsilon) {
        // This is a corner case when using decaying target calcium
        if (current == eta) {
            return 0.0;
        }
        return -1.0;
    }

    constexpr auto factor = 1.6651092223153955127063292897904020952611777045288814583336582344;
    constexpr auto factor_inv = 1.0 / factor;
    // 1.6651092223153955127063292897904020952611777045288814583336582344... = (2 * sqrt(-log(0.5)))

    const auto xi = (eta + epsilon) * 0.5;
    const auto zeta = (eta - epsilon) * factor_inv;

    const auto difference = current - xi;
    const auto quotient = difference / zeta;
    const auto product = quotient * quotient;

    const auto dz = (2 * std::exp(-product)) - 1;
    return dz;
}
