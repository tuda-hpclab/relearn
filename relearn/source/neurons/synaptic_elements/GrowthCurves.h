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

#include "types/BasicTypes.h"

#include <cpp-utility/Cast.hpp>

#include <cmath>

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
inline RelearnTypes::grown_type gaussian_growth_curve(const RelearnTypes::calcium_type current, const RelearnTypes::calcium_type eta, const RelearnTypes::calcium_type epsilon) noexcept {
    using calcium_type = RelearnTypes::calcium_type;

    if (eta == epsilon) {
        // This is a corner case when using decaying target calcium
        if (current == eta) {
            return RelearnTypes::grown_type{ 0 };
        }
        return RelearnTypes::grown_type{ -1 };
    }

    constexpr auto factor = utility::as<calcium_type>(1.6651092223153955127063292897904020952611777045288814583336582344);
    constexpr auto factor_inv = calcium_type{ 1 } / factor;
    // 1.6651092223153955127063292897904020952611777045288814583336582344... = (2 * sqrt(-log(0.5)))

    const auto xi = (eta + epsilon) * calcium_type{ 0.5 };
    const auto zeta = (eta - epsilon) * factor_inv;

    const auto difference = current - xi;
    const auto quotient = difference / zeta;
    const auto product = quotient * quotient;

    const auto dz = (calcium_type{ 2 } * std::exp(-product)) - calcium_type{ 1 };
    return dz;
}
