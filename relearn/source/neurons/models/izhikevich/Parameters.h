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

namespace models::izhikevich {
/**
 * This class contains the parameters for the Izhikevich model.
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
     * @param _a The dampening factor for u(t)
     * @param _b The dampening factor for v(t) inside the equation for d/dt u(t)
     * @param _c The reset activity
     * @param _d The additional dampening for u(t) in case of spiking
     * @param _V_spike The spiking threshold
     * @param _k1 The factor for v(t)^2 inside the equation for d/dt v(t)
     * @param _k2 The factor for v(t) inside the equation for d/dt v(t)
     * @param _k3 The constant inside the equation for d/dt v(t)
     */
    Parameters(
        const real_type _a,
        const real_type _b,
        const real_type _c,
        const real_type _d,
        const real_type _V_spike,
        const real_type _k1,
        const real_type _k2,
        const real_type _k3) noexcept
        : a{ _a }
        , b{ _b }
        , c{ _c }
        , d{ _d }
        , V_spike{ _V_spike }
        , k1{ _k1 }
        , k2{ _k2 }
        , k3{ _k3 } { }

    /**
     * @brief Returns the linear scaling term of the u(t) equation
     * @return a
     */
    [[nodiscard]] real_type get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns the linear scaling term of v(t) in the u(t) equation
     * @return b
     */
    [[nodiscard]] real_type get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns the reset value of v(t) in case of spiking
     * @return c
     */
    [[nodiscard]] real_type get_c() const noexcept {
        return c;
    }

    /**
     * @brief Returns the increase of u(t) in case of spiking
     * @return d
     */
    [[nodiscard]] real_type get_d() const noexcept {
        return d;
    }

    /**
     * @brief Returns the spike threshold for v(t)
     * @return The threshold
     */
    [[nodiscard]] real_type get_V_spike() const noexcept {
        return V_spike;
    }

    /**
     * @brief Returns the scaling factor of v(t)^2 in the v(t) equation
     * @return k1
     */
    [[nodiscard]] real_type get_k1() const noexcept {
        return k1;
    }

    /**
     * @brief Returns the scaling factor of v(t) in the v(t) equation
     * @return k2
     */
    [[nodiscard]] real_type get_k2() const noexcept {
        return k2;
    }

    /**
     * @brief Returns the constant term in the v(t) equation
     * @return k3
     */
    [[nodiscard]] real_type get_k3() const noexcept {
        return k3;
    }

    static constexpr real_type default_a{ 0.1 };
    static constexpr real_type default_b{ 0.2 };
    static constexpr real_type default_c{ -65.0 };
    static constexpr real_type default_d{ 2.0 };
    static constexpr real_type default_V_spike{ 30.0 };
    static constexpr real_type default_k1{ 0.04 };
    static constexpr real_type default_k2{ 5.0 };
    static constexpr real_type default_k3{ 140.0 };

    static constexpr real_type min_a{ 0.0 };
    static constexpr real_type min_b{ 0.0 };
    static constexpr real_type min_c{ -150.0 };
    static constexpr real_type min_d{ 0.0 };
    static constexpr real_type min_V_spike{ 0.0 };
    static constexpr real_type min_k1{ 0.0 };
    static constexpr real_type min_k2{ 0.0 };
    static constexpr real_type min_k3{ 50.0 };

    static constexpr real_type max_a{ 1.0 };
    static constexpr real_type max_b{ 1.0 };
    static constexpr real_type max_c{ -50.0 };
    static constexpr real_type max_d{ 10.0 };
    static constexpr real_type max_V_spike{ 100.0 };
    static constexpr real_type max_k1{ 1.0 };
    static constexpr real_type max_k2{ 10.0 };
    static constexpr real_type max_k3{ 200.0 };

private:
    real_type a{ default_a };
    real_type b{ default_b };
    real_type c{ default_c };
    real_type d{ default_d };

    real_type V_spike{ default_V_spike };

    real_type k1{ default_k1 };
    real_type k2{ default_k2 };
    real_type k3{ default_k3 };
};

} // namespace models::izhikevich
