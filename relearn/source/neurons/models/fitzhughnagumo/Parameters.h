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

namespace models::fitzhughnagumo {

/**
 * This class contains the parameters for the FitzHugh-Nagumo model.
 * @tparam real_type The type of the parameters, i.e., of the membrane potential and the recovery variables
 */
template <typename real_type = RelearnTypes::activity_type>
class Parameters {
public:
    /**
     * @brief Initializes the parameters with the default values.
     */
    Parameters() = default;

    /**
     * @brief Initializes the parameters with the passed values.
     * @param _a The additive constant inside the equation for d/dt w(t)
     * @param _b The linear scaling term of w(t) inside the equation for d/dt w(t)
     * @param _phi The linear scaling term inside the equation for d/dt w(t)
     * @param _init_x The initial value for x
     * @param _init_w The initial value for w
     */
    Parameters(
        const real_type _a,
        const real_type _b,
        const real_type _phi,
        const real_type _init_x,
        const real_type _init_w) noexcept
        : a{ _a }
        , b{ _b }
        , phi{ _phi }
        , init_x{ _init_x }
        , init_w{ _init_w } { }

    /**
     * @brief Returns the additive constant inside the equation for d/dt w(t)
     * @return a
     */
    [[nodiscard]] real_type get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns the linear scaling term of w(t) inside the equation for d/dt w(t)
     * @return b
     */
    [[nodiscard]] real_type get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns the linear scaling term inside the equation for d/dt w(t)
     * @return phi
     */
    [[nodiscard]] real_type get_phi() const noexcept {
        return phi;
    }

    /**
     * @brief Returns the initial value for x
     * @return init_x
     */
    [[nodiscard]] real_type get_init_x() const noexcept {
        return init_x;
    }

    /**
     * @brief Returns the initial value for w
     * @return init_w
     */
    [[nodiscard]] real_type get_init_w() const noexcept {
        return init_w;
    }

    static constexpr real_type default_a{ utility::as<real_type>(0.7) };
    static constexpr real_type default_b{ utility::as<real_type>(0.8) };
    static constexpr real_type default_phi{ utility::as<real_type>(0.08) };
    static constexpr real_type default_init_x{ utility::as<real_type>(-1.2) };
    static constexpr real_type default_init_w{ utility::as<real_type>(-0.6) };

    static constexpr real_type min_a{ utility::as<real_type>(0.6) };
    static constexpr real_type min_b{ utility::as<real_type>(0.7) };
    static constexpr real_type min_phi{ utility::as<real_type>(0.07) };
    static constexpr real_type min_init_x{ utility::as<real_type>(-200.0) };
    static constexpr real_type min_init_w{ utility::as<real_type>(-100.0) };

    static constexpr real_type max_a{ utility::as<real_type>(0.8) };
    static constexpr real_type max_b{ utility::as<real_type>(0.9) };
    static constexpr real_type max_phi{ utility::as<real_type>(0.09) };
    static constexpr real_type max_init_x{ utility::as<real_type>(100.0) };
    static constexpr real_type max_init_w{ utility::as<real_type>(20.0) };

private:
    real_type a{ default_a };
    real_type b{ default_b };
    real_type phi{ default_phi };

    real_type init_x{ default_init_x };
    real_type init_w{ default_init_w };
};

} // namespace models::fitzhughnagumo
