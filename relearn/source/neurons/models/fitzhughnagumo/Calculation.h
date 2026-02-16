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

namespace models::fitzhughnagumo {

template <typename real_type = double>
[[nodiscard]] real_type iterate_v(const real_type current_v, const real_type current_w, const real_type input) noexcept {
    return current_v - (current_v * current_v * current_v / 3.0) - current_w + input;
}

template <typename real_type = double>
[[nodiscard]] real_type iterate_w(const real_type current_v, const real_type current_w, const real_type a, const real_type b, const real_type phi) noexcept {
    return phi * (current_v + a - b * current_w);
}

template <typename real_type = double>
[[nodiscard]] bool is_spiking(const real_type current_v, const real_type /*current_w*/, const real_type V_spike) noexcept {
    return current_v >= V_spike;
}

template <typename real_type = double>
[[nodiscard]] std::tuple<real_type, real_type, bool> iterate(const unsigned int h, const real_type current_v, const real_type current_w,
                                                             const real_type input, const Parameters<real_type>& params) noexcept {

    const auto a = params.get_a();
    const auto b = params.get_b();
    const auto phi = params.get_phi();

    auto v = current_v;
    auto w = current_w;

    const auto scale = real_type(1.0) / h;

    for (auto i = 0U; i < h; ++i) {
        const auto v_increase = v - (v * v * v / 3.0) - w + input;
        const auto w_increase = phi * (v + a - b * w);

        v += scale * v_increase;
        w += scale * w_increase;
    }

    const auto spiked = w > v - v * v * v / 3.0 && v > 1.0;
    return { v, w, spiked };
}

} // namespace models::fitzhughnagumo
