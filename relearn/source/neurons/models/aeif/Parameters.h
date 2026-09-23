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

namespace models::aeif {
/**
 * This class contains the parameters for the AEIF model.
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
     * @param _C Membrane capacitance
     * @param _g_L Leak conductance
     * @param _E_L Leak reversal potential
     * @param _V_T Spike threshold
     * @param _d_T Slope factor
     * @param _tau_w Adaptation time constant
     * @param _a Sub-threshold adaptation
     * @param _b Spike-triggered adaptation
     * @param _V_spike Spike trigger
     */
    Parameters(
        const real_type _C,
        const real_type _g_L,
        const real_type _E_L,
        const real_type _V_T,
        const real_type _d_T,
        const real_type _tau_w,
        const real_type _a,
        const real_type _b,
        const real_type _V_spike) noexcept
        : C{ _C }
        , g_L{ _g_L }
        , E_L{ _E_L }
        , V_T{ _V_T }
        , d_T{ _d_T }
        , tau_w{ _tau_w }
        , a{ _a }
        , b{ _b }
        , V_spike{ _V_spike } { }

    /**
     * @brief Returns the membrane capacitance
     * @return C
     */
    [[nodiscard]] real_type get_C() const noexcept {
        return C;
    }

    /**
     * @brief Returns the leak conductance
     * @return g_L
     */
    [[nodiscard]] real_type get_g_L() const noexcept {
        return g_L;
    }

    /**
     * @brief Returns the leak reversal potential
     * @return E_L
     */
    [[nodiscard]] real_type get_E_L() const noexcept {
        return E_L;
    }

    /**
     * @brief Returns the spike threshold
     * @return V_T
     */
    [[nodiscard]] real_type get_V_T() const noexcept {
        return V_T;
    }

    /**
     * @brief Returns the slope factor
     * @return d_T
     */
    [[nodiscard]] real_type get_d_T() const noexcept {
        return d_T;
    }

    /**
     * @brief Returns the adaptation time constant
     * @return tau_w
     */
    [[nodiscard]] real_type get_tau_w() const noexcept {
        return tau_w;
    }

    /**
     * @brief Returns the sub-threshold adaptation
     * @return a
     */
    [[nodiscard]] real_type get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns the spike-triggered adaptation
     * @return b
     */
    [[nodiscard]] real_type get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns the spike trigger
     * @return V_spike
     */
    [[nodiscard]] real_type get_V_spike() const noexcept {
        return V_spike;
    }

    static constexpr real_type default_C{ utility::as<real_type>(281.0) };
    static constexpr real_type default_g_L{ utility::as<real_type>(30.0) };
    static constexpr real_type default_E_L{ utility::as<real_type>(-70.6) };
    static constexpr real_type default_V_T{ utility::as<real_type>(-50.4) };
    static constexpr real_type default_d_T{ utility::as<real_type>(2.0) };
    static constexpr real_type default_tau_w{ utility::as<real_type>(144.0) };
    static constexpr real_type default_a{ utility::as<real_type>(4.0) };
    static constexpr real_type default_b{ utility::as<real_type>(0.0805) };
    static constexpr real_type default_V_spike{ utility::as<real_type>(20.0) };

    static constexpr real_type min_C{ utility::as<real_type>(100.0) };
    static constexpr real_type min_g_L{ utility::as<real_type>(0.0) };
    static constexpr real_type min_E_L{ utility::as<real_type>(-150.0) };
    static constexpr real_type min_V_T{ utility::as<real_type>(-150.0) };
    static constexpr real_type min_d_T{ utility::as<real_type>(0.0) };
    static constexpr real_type min_tau_w{ utility::as<real_type>(100.0) };
    static constexpr real_type min_a{ utility::as<real_type>(0.0) };
    static constexpr real_type min_b{ utility::as<real_type>(0.0) };
    static constexpr real_type min_V_spike{ utility::as<real_type>(0.0) };

    static constexpr real_type max_C{ utility::as<real_type>(500.0) };
    static constexpr real_type max_g_L{ utility::as<real_type>(100.0) };
    static constexpr real_type max_E_L{ utility::as<real_type>(-20.0) };
    static constexpr real_type max_V_T{ utility::as<real_type>(0.0) };
    static constexpr real_type max_d_T{ utility::as<real_type>(10.0) };
    static constexpr real_type max_tau_w{ utility::as<real_type>(200.0) };
    static constexpr real_type max_a{ utility::as<real_type>(10.0) };
    static constexpr real_type max_b{ utility::as<real_type>(0.3) };
    static constexpr real_type max_V_spike{ utility::as<real_type>(70.0) };

private:
    real_type C{ default_C };             // membrane capacitance
    real_type g_L{ default_g_L };         // leak conductance
    real_type E_L{ default_E_L };         // leak reversal potential
    real_type V_T{ default_V_T };         // spike threshold
    real_type d_T{ default_d_T };         // slope factor
    real_type tau_w{ default_tau_w };     // adaptation time constant
    real_type a{ default_a };             // sub-threshold
    real_type b{ default_b };             // spike-triggered adaptation
    real_type V_spike{ default_V_spike }; // spike trigger
};

} // namespace models::aeif
