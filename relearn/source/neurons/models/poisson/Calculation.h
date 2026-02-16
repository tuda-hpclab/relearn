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

#include "Parameters.h"

#include <tuple>

namespace models::poisson {

template <typename real_type = double>
[[nodiscard]] real_type iterate_x(const real_type current_x, const real_type input, const real_type x_0, const real_type tau_x_inverse) noexcept {
    return ((x_0 - current_x) * tau_x_inverse) + input;
}

template <typename real_type = double>
[[nodiscard]] bool is_spiking(const real_type current_x, const real_type threshold) noexcept {
    return current_x >= threshold;
}

template <typename real_type = double, typename integral_type = unsigned int>
[[nodiscard]] std::tuple<real_type, integral_type, bool> iterate(const unsigned int h, const real_type current_x, const integral_type current_refrac,
                                                                 const real_type input, const real_type threshold, const Parameters<real_type, integral_type>& params) noexcept {
    const auto x_0 = params.get_x_0();
    const auto tau_x = params.get_tau_x();
    const auto refractory_time = params.get_refractory_period();

    auto x = current_x;

    const auto scale = real_type(1.0) / h;
    const auto tau_x_inverse = real_type(1.0) / tau_x;

    for (auto i = 0U; i < h; ++i) {
        const auto x_increase = ((x_0 - x) * tau_x_inverse) + input;
        x += scale * x_increase;
    }

    if (current_refrac > 0) {
        return { x, current_refrac - 1, false };
    }

    if (x >= threshold) {
        return { x, refractory_time, true };
    }

    return { x, 0, false };
}

} // namespace models::poisson
