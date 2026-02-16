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

namespace models::poisson {

/**
 * This class contains the parameters for the Poisson model.
 * @tparam real_type The type of the parameters, default is double
 */
template <typename real_type = double, typename integral_type = unsigned int>
class Parameters {
public:
    /**
     * @brief Initializes the parameters with the default values.
     */
    Parameters() = default;

    /**
     * @brief Initializes the parameters with the passed values.
     * @param _x_0 The resting membrane potential
     * @param _tau_x The dampening factor by which the membrane potential decreases
     * @param _refractory_period The number of steps a neuron doesn't spike after spiking
     */
    Parameters(
        const real_type _x_0,
        const real_type _tau_x,
        const integral_type _refractory_period) noexcept
        : x_0{ _x_0 }
        , tau_x{ _tau_x }
        , refractory_period{ _refractory_period } { }

    /**
     * @brief Returns the resting membrane potential
     * @return x_0
     */
    [[nodiscard]] real_type get_x_0() const noexcept {
        return x_0;
    }

    /**
     * @brief Returns the dampening factor by which the membrane potential decreases
     * @return tau_x
     */
    [[nodiscard]] real_type get_tau_x() const noexcept {
        return tau_x;
    }

    /**
     * @brief Returns the refractory period
     * @return refractory_period
     */
    [[nodiscard]] integral_type get_refractory_period() const noexcept {
        return refractory_period;
    }

    static constexpr real_type default_x_0{ 0.05 };
    static constexpr real_type default_tau_x{ 5.0 };
    static constexpr integral_type default_refractory_period{ 4 };

    static constexpr real_type min_x_0{ 0.0 };
    static constexpr real_type min_tau_x{ 0.0 };
    static constexpr integral_type min_refractory_time{ 0 };

    static constexpr real_type max_x_0{ 1.0 };
    static constexpr real_type max_tau_x{ 1000.0 };
    static constexpr integral_type max_refractory_time{ 1000 };

private:
    // Background or resting activity
    real_type x_0{ default_x_0 };

    // Decay time of firing rate in msec
    real_type tau_x{ default_tau_x };

    // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
    integral_type refractory_period{ default_refractory_period };
};

} // namespace models::poisson
