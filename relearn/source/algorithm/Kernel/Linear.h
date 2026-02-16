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

/**
 * Offers an inheritance-based interface to calculate the attraction linearly with
 * a cut-off, i.e., the attraction is 0 if the distance is larger then the cut-off,
 * constant if the cut-off is infinite, and linear interpolated based on the
 * distance is neither is the case
 */
class LinearDistributionKernel : public KernelBase {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    using KernelBase::get_probability;

    static constexpr double default_cutoff = std::numeric_limits<double>::infinity();

    /**
     * @brief Constructs a new Gaussian kernel
     * @param _cutoff The variance sigma, must be >= 0.0
     * @exception Throws a RelearnException if _cutoff < 0.0
     */
    explicit LinearDistributionKernel(const double _cutoff = default_cutoff)
        : cutoff{ _cutoff } {
        RelearnException::check(_cutoff >= 0.0, "LinearDistributionKernel::LinearDistributionKernel, _cutoff was less than 0.0");
    }

    ~LinearDistributionKernel() override = default;

    /**
     * @brief Returns the currently used cut-off parameter
     * @return The currently used cut-off parameter
     */
    [[nodiscard]] double get_cutoff() const noexcept {
        return cutoff;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of ||s - t||_2,
     *      i.e., if this is smaller than the cut-off point, the return value is k, otherwise it's 0.0
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] double get_probability(const double distance) const override {
        if (std::isinf(cutoff)) {
            return 1.0;
        }

        if (distance > cutoff) {
            return 0.0;
        }

        const auto factor = distance / cutoff;
        return 1 - factor;
    }

    [[nodiscard]] bool is_approximately_equal(const KernelBase& other, double epsilon = Constants::eps) const override {
        const auto double_equal = [epsilon](double a, double b) {
            return fabs(a - b) < epsilon;
        };
        const auto* other_kernel = dynamic_cast<const LinearDistributionKernel*>(&other);
        if (!other_kernel) {
            return false;
        }
        return double_equal(get_cutoff(), other_kernel->get_cutoff());
    }

    [[nodiscard]] KernelType get_kernel_type() const override {
        return KernelType::Linear;
    }

    [[nodiscard]] std::string to_humanreadable_string() const override {
        return fmt::format("Kernel Linear with cutoff={}", cutoff);
    }

private:
    double cutoff{ default_cutoff };
};
