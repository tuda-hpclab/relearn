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

/**
 * Offers an inheritance-based interface to calculate the attraction linearly with
 * a cut-off, i.e., the attraction is 0 if the distance is larger then the cut-off,
 * constant if the cut-off is infinite, and linear interpolated based on the
 * distance is neither is the case
 */
class LinearDistributionKernel : public KernelBase {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;
    using space_type = RelearnTypes::space_type;

    using KernelBase::get_probability;

    static constexpr attraction_type default_cutoff = std::numeric_limits<attraction_type>::infinity();

    /**
     * @brief Constructs a new Gaussian kernel
     * @param _cutoff The variance sigma, must be >= 0.0
     * @exception Throws a RelearnException if _cutoff < 0.0
     */
    explicit LinearDistributionKernel(const attraction_type _cutoff = default_cutoff)
        : cutoff{ _cutoff } {
        RelearnException::check(_cutoff >= attraction_type{ 0 }, "LinearDistributionKernel::LinearDistributionKernel, _cutoff was less than 0.0");
    }

    ~LinearDistributionKernel() override = default;

    /**
     * @brief Returns the currently used cut-off parameter
     * @return The currently used cut-off parameter
     */
    [[nodiscard]] attraction_type get_cutoff() const noexcept {
        return cutoff;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of ||s - t||_2,
     *      i.e., if this is smaller than the cut-off point, the return value is k, otherwise it's 0.0
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] attraction_type get_probability(const space_type distance) const override {
        if (std::isinf(cutoff)) {
            return 1.0;
        }

        if (distance > cutoff) {
            return 0.0;
        }

        const auto factor = distance / cutoff;
        return 1 - factor;
    }

    [[nodiscard]] bool is_approximately_equal(const KernelBase& other, attraction_type epsilon = static_cast<attraction_type>(Constants::eps)) const override {
        const auto values_equal = [epsilon](const attraction_type a, const attraction_type b) {
            return std::fabs(a - b) < epsilon;
        };
        const auto* other_kernel = dynamic_cast<const LinearDistributionKernel*>(&other);
        if (!other_kernel) {
            return false;
        }
        return values_equal(get_cutoff(), other_kernel->get_cutoff());
    }

    [[nodiscard]] KernelType get_kernel_type() const override {
        return KernelType::Linear;
    }

    [[nodiscard]] std::string to_humanreadable_string() const override {
        return fmt::format("Kernel Linear with cutoff={}", cutoff);
    }

private:
    attraction_type cutoff{ default_cutoff };
};
