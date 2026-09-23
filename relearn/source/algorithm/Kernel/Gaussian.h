#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "algorithm/Kernel/KernelBase.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cmath>
#include <numeric>

/**
 * Offers an inheritance-based interface to calculate the attraction based on a gaussian, i.e.,
 * if x is the distance, the attraction is proportional to
 * exp(-((x-mu)/sigma)^2)
 */
class GaussianDistributionKernel : public KernelBase {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;
    using space_type = RelearnTypes::space_type;

    using KernelBase::get_probability;

    static constexpr attraction_type default_mu = 0.0;      // In Sebastian's work: 0.0
    static constexpr attraction_type default_sigma = 750.0; // In Sebastian's work: 750.0

    /**
     * @brief Constructs a new Gaussian kernel
     * @param _mu The offset mu
     * @param _sigma The variance sigma, must be > 0.0
     * @exception Throws a RelearnException if sigma <= 0.0
     */
    GaussianDistributionKernel(const attraction_type _mu = default_mu, const attraction_type _sigma = default_sigma)
        : mu{ _mu }
        , sigma{ _sigma } {
        RelearnException::check(sigma > attraction_type{ 0 }, "GaussianDistributionKernel::GaussianDistributionKernel, sigma was not greater than 0.0");
        scale_factor = attraction_type{ -1 } / (sigma * sigma);
    }

    ~GaussianDistributionKernel() override = default;

    /**
     * @brief Calculates the probability for a connection based on the distance on the basis of
     *      exp(-||s - t||_2^2 / sigma^2)
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] attraction_type get_probability(const space_type distance) const override {
        // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
        const auto numerator = (distance - mu) * (distance - mu);
        const auto exponent = numerator * scale_factor;
        const auto exp_val = std::exp(exponent);
        return exp_val;
    }

    /**
     * @brief Returns the currently used offset
     * @return The currently used offset
     */
    [[nodiscard]] attraction_type get_mu() const noexcept {
        return mu;
    }

    /**
     * @brief Returns the currently used variance
     * @return The currently used variance
     */
    [[nodiscard]] attraction_type get_sigma() const noexcept {
        return sigma;
    }

    [[nodiscard]] bool is_approximately_equal(const KernelBase& other, attraction_type epsilon = static_cast<attraction_type>(Constants::eps)) const override {
        const auto values_equal = [epsilon](const attraction_type a, const attraction_type b) {
            return std::fabs(a - b) < epsilon;
        };
        const auto* other_kernel = dynamic_cast<const GaussianDistributionKernel*>(&other);
        if (!other_kernel) {
            return false;
        }
        return values_equal(get_mu(), other_kernel->get_mu()) && values_equal(get_sigma(), other_kernel->get_sigma());
    }

    [[nodiscard]] KernelType get_kernel_type() const override {
        return KernelType::Gaussian;
    }

    std::string to_humanreadable_string() const override {
        return fmt::format("Kernel Gaussian with mu={} sigma={}", mu, sigma);
    }

private:
    attraction_type mu{ default_mu };
    attraction_type sigma{ default_sigma };

    attraction_type scale_factor{ attraction_type{ -1 } / (default_sigma * default_sigma) };
};
