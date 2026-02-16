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

namespace models::fitzhughnagumo {

/**
 * This class contains the parameters for the FitzHugh-Nagumo model.
 * @tparam real_type The type of the parameters, default is double
 */
template <typename real_type = double>
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

    static constexpr real_type default_a{ 0.7 };
    static constexpr real_type default_b{ 0.8 };
    static constexpr real_type default_phi{ 0.08 };
    static constexpr real_type default_init_x{ -1.2 };
    static constexpr real_type default_init_w{ -0.6 };

    static constexpr real_type min_a{ 0.6 };
    static constexpr real_type min_b{ 0.7 };
    static constexpr real_type min_phi{ 0.07 };
    static constexpr real_type min_init_x{ -200.0 };
    static constexpr real_type min_init_w{ -100.0 };

    static constexpr real_type max_a{ 0.8 };
    static constexpr real_type max_b{ 0.9 };
    static constexpr real_type max_phi{ 0.09 };
    static constexpr real_type max_init_x{ 100.0 };
    static constexpr real_type max_init_w{ 20.0 };

private:
    real_type a{ default_a };
    real_type b{ default_b };
    real_type phi{ default_phi };

    real_type init_x{ default_init_x };
    real_type init_w{ default_init_w };
};

} // namespace models::fitzhughnagumo
