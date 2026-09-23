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

#include "Parameters.h"

#include <cmath>
#include <tuple>

namespace models::aeif {

template <typename real_type = RelearnTypes::activity_type>
[[nodiscard]] real_type iterate_v(const real_type current_v, const real_type current_w, const real_type input, const real_type g_L, const real_type E_L,
                                  const real_type d_T, const real_type V_T, const real_type C_inverse) noexcept {
    const auto d_T_inverse = 1.0 / d_T;
    const auto linear_part = -g_L * (current_v - E_L);
    const auto exp_part = g_L * d_T * std::exp((current_v - V_T) * d_T_inverse);

    return (linear_part + exp_part - current_w + input) * C_inverse;
}

template <typename real_type = RelearnTypes::activity_type>
[[nodiscard]] real_type iterate_w(const real_type current_v, const real_type current_w, const real_type a, const real_type E_L, const real_type tau_W_inverse) noexcept {
    return (a * (current_v - E_L) - current_w) * tau_W_inverse;
}

template <typename real_type = RelearnTypes::activity_type>
[[nodiscard]] bool is_spiking(const real_type current_v, const real_type V_spike) noexcept {
    return current_v >= V_spike;
}

template <typename real_type = RelearnTypes::activity_type>
[[nodiscard]] std::tuple<real_type, real_type, bool> iterate(const unsigned int h, const real_type current_v, const real_type current_w,
                                                             const real_type input, const Parameters<real_type>& params) noexcept {

    const auto d_T = params.get_d_T();
    const auto C = params.get_C();
    const auto tau_w = params.get_tau_w();

    const auto d_T_inverse = 1.0 / d_T;
    const auto C_inverse = 1.0 / C;
    const auto tau_w_inverse = 1.0 / tau_w;

    const auto g_L = params.get_g_L();
    const auto E_L = params.get_E_L();
    const auto V_T = params.get_V_T();
    const auto a = params.get_a();
    const auto b = params.get_b();
    const auto V_spike = params.get_V_spike();

    auto v = current_v;
    auto w = current_w;

    const auto scale = 1.0 / h;

    for (auto integration_steps = 0U; integration_steps < h; ++integration_steps) {
        const auto linear_part = -g_L * (v - E_L);
        const auto exp_part = g_L * d_T * std::exp((v - V_T) * d_T_inverse);
        const auto x_increase = (linear_part + exp_part - w + input) * C_inverse;
        const auto w_increase = (a * (v - E_L) - w) * tau_w_inverse;

        v += x_increase * scale;
        w += w_increase * scale;

        if (v >= V_spike) {
            return { E_L, w + b, true };
        }
    }

    return { v, w, false };
}

} // namespace models::aeif
