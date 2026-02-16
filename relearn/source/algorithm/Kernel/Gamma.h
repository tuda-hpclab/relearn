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

#include "Config.h"
#include "Types.h"

#include "algorithm/Kernel/KernelBase.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cmath>
#include <numeric>

/**
 * Offers an inheritance-based interface to calculate the attraction based on a gamma distribution, i.e.,
 * if x is the distance, the attraction is proportional to
 * 1/[Gamma(k)*theta^k] * x^(k-1) * exp(-x/theta)
 */
class GammaDistributionKernel : public KernelBase {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    using KernelBase::get_probability;

    static constexpr double default_k = 1.0;
    static constexpr double default_theta = 1.0;

    /**
     * @brief Constructs a new Gamma kernel
     * @param _k The shape parameter, > 0.0
     * @param _theta The scale parameter, > 0.0
     * @exception Throws a RelearnException if sigma <= 0.0
     */
    explicit GammaDistributionKernel(const double _k = default_k, const double _theta = default_theta)
        : k{ _k }
        , theta{ _theta } {
        RelearnException::check(_k > 0.0, "GammaDistributionKernel::GammaDistributionKernel, k was not greater than 0.0");
        RelearnException::check(_theta > 0.0, "GammaDistributionKernel::GammaDistributionKernel, k was not greater than 0.0");

        gamma_divisor_inv = 1.0 / (std::tgamma(_k) * std::pow(_theta, _k));
        theta_divisor = -1.0 / _theta;
    }

    ~GammaDistributionKernel() override = default;

    /**
     * @brief Returns the currently used shape parameter
     * @return The currently used shape parameter
     */
    [[nodiscard]] double get_k() const noexcept {
        return k;
    }

    /**
     * @brief Returns the currently used scale parameter
     * @return The currently used scale parameter
     */
    [[nodiscard]] double get_theta() const noexcept {
        return theta;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the gamma distribution
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] double get_probability(const double distance) const override {
        const auto factor_1 = std::pow(distance, k - 1);
        const auto factor_2 = std::exp(distance * theta_divisor);

        const auto result = gamma_divisor_inv * factor_1 * factor_2;

        return result;
    }

    [[nodiscard]] bool is_approximately_equal(const KernelBase& other, double epsilon = Constants::eps) const override {
        const auto double_equal = [epsilon](double a, double b) {
            return fabs(a - b) < epsilon;
        };

        const auto* other_kernel = dynamic_cast<const GammaDistributionKernel*>(&other);
        if (!other_kernel) {
            return false;
        }
        return double_equal(get_k(), other_kernel->get_k()) && double_equal(get_theta(), other_kernel->get_theta());
    }

    [[nodiscard]] KernelType get_kernel_type() const override {
        return KernelType::Gamma;
    }

    std::string to_humanreadable_string() const override {
        return fmt::format("Kernel Gamma with k={}", k);
    };

private:
    double k{ default_k };
    double theta{ default_theta };

    double gamma_divisor_inv{ 1.0 / (std::tgamma(k) * std::pow(theta, k)) };
    double theta_divisor{ -1.0 / theta };
};
