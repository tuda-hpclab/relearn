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

namespace models::izhikevich {

template <typename real_type = double>
[[nodiscard]] real_type iterate_v(const real_type current_v, const real_type current_u, const real_type input, const real_type k2, const real_type k1, const real_type k0) noexcept {
    return (k2 * current_v * current_v) + (k1 * current_v) + k0 - current_u + input;
}

template <typename real_type = double>
[[nodiscard]] real_type iterate_u(const real_type current_v, const real_type current_u, const real_type a, const real_type b) noexcept {
    return a * (b * current_v - current_u);
}

template <typename real_type = double>
[[nodiscard]] bool is_spiking(const real_type current_v, const real_type V_spike) noexcept {
    return current_v >= V_spike;
}

template <typename real_type = double>
[[nodiscard]] std::tuple<real_type, real_type, bool> iterate(const unsigned int h, const real_type current_v, const real_type current_u,
                                                             const real_type input, const Parameters<real_type>& params) noexcept {
    const auto k2 = params.get_k1();
    const auto k1 = params.get_k2();
    const auto k0 = params.get_k3();
    const auto a = params.get_a();
    const auto b = params.get_b();
    const auto c = params.get_c();
    const auto d = params.get_d();
    const auto V_spike = params.get_V_spike();

    auto v = current_v;
    auto u = current_u;

    const auto scale = real_type(1.0) / h;

    for (auto i = 0U; i < h; ++i) {
        const auto v_increase = (k2 * v * v) + (k1 * v) + k0 - u + input;
        const auto u_increase = a * (b * v - u);

        v += scale * v_increase;
        u += scale * u_increase;

        if (v >= V_spike) {
            return { c, u + d, true };
        }
    }

    return { v, u, false };
}

} // namespace models::izhikevich
