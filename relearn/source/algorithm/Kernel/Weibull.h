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
 * Offers an inheritance-based interface to calculate the attraction based on a weibull distribution, i.e.,
 * if x is the distance, the attraction is proportional to
 * b * k * x^(k-1) * exp(-b * x^k)
 */
class WeibullDistributionKernel : public KernelBase {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;
    using space_type = RelearnTypes::space_type;

    using KernelBase::get_probability;

    static constexpr attraction_type default_k = 1.0;
    static constexpr attraction_type default_b = 1.0;

    /**
     * @brief Constructs a new Gaussian kernel
     * @param _k The shape parameter, > 0.0
     * @param _b The scaling parameter, > 0.0
     * @exception Throws a RelearnException if _k <= 0.0 or _b <= 0.0
     */
    WeibullDistributionKernel(const attraction_type _k = default_k, const attraction_type _b = default_b)
        : k{ _k }
        , b{ _b } {
        RelearnException::check(_k > attraction_type{ 0 }, "WeibullDistributionKernel::WeibullDistributionKernel, _k was not greater than 0.0");
        RelearnException::check(_b > attraction_type{ 0 }, "WeibullDistributionKernel::WeibullDistributionKernel, _b was not greater than 0.0");
    }

    ~WeibullDistributionKernel() override = default;

    /**
     * @brief Returns the currently used shape parameter
     * @return The currently used shape parameter
     */
    [[nodiscard]] attraction_type get_k() const noexcept {
        return k;
    }

    /**
     * @brief Returns the currently used scale parameter
     * @return The currently used scale parameter
     */
    [[nodiscard]] attraction_type get_b() const noexcept {
        return b;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the weibull distribution
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] attraction_type get_probability(const space_type distance) const override {
        const auto factor_1 = b * k;

        const auto factor_2 = std::pow(distance, k - 1);
        const auto exponent = -b * factor_2 * distance;
        const auto factor_3 = std::exp(exponent);

        const auto result = factor_1 * factor_2 * factor_3;

        return result;
    }

    [[nodiscard]] bool is_approximately_equal(const KernelBase& other, attraction_type epsilon = static_cast<attraction_type>(Constants::eps)) const override {
        const auto values_equal = [epsilon](const attraction_type first, const attraction_type second) {
            return std::fabs(first - second) < epsilon;
        };
        const auto* other_kernel = dynamic_cast<const WeibullDistributionKernel*>(&other);
        if (!other_kernel) {
            return false;
        }
        return values_equal(get_b(), other_kernel->get_b()) && values_equal(get_k(), other_kernel->get_k());
    }

    [[nodiscard]] KernelType get_kernel_type() const override {
        return KernelType::Weibull;
    }

    std::string to_humanreadable_string() const override {
        return fmt::format("Kernel Weibull with k={} b={}", k, b);
    }

private:
    attraction_type k{ default_k };
    attraction_type b{ default_b };
};
