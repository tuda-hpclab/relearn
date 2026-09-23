#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "algorithm/Kernel/KernelType.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"

class KernelBase {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;
    using space_type = RelearnTypes::space_type;

    KernelBase() = default;
    virtual ~KernelBase() = default;

    KernelBase(const KernelBase&) = default;
    KernelBase& operator=(const KernelBase&) = default;

    KernelBase(KernelBase&&) = default;
    KernelBase& operator=(KernelBase&&) = default;

    /**
     * @brief Calculates the probability for a connection based on the distance
     * @param distance The distance between the source and target neuron
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] virtual attraction_type get_probability(space_type distance) const = 0;

    /**
     * @brief Calculates the probability for a connection based on the positions of the source and target neuron
     * @param source_position The position of the source neuron
     * @param target_position The position of the target neuron
     * @param number_free_target_elements The number of free target elements
     * @return The probability for a connection, >= 0.0; not normalized to [0, 1]
     */
    [[nodiscard]] attraction_type get_probability(const position_type& source_position, const position_type& target_position, const counter_type number_free_target_elements = 1) const {
        // Note: This function is here to avoid code duplication in the derived classes
        const auto position_diff = target_position - source_position;
        const auto distance = position_diff.calculate_2_norm<space_type>();
        return get_probability(distance) * static_cast<attraction_type>(number_free_target_elements);
    }

    [[nodiscard]] virtual KernelType get_kernel_type() const = 0;

    /**
     * @brief Checks whether this kernel is approximately equal to the other one. This is not transitive! It accounts for
     *        small deviations in the parameters by using Constants::eps
     * @param other The other kernel
     * @param epsilon The epsilon used to compare the parameters
     * @return Whether this kernel is approximately equal to the other one
     */
    [[nodiscard]] virtual bool is_approximately_equal(const KernelBase& other, attraction_type epsilon = static_cast<attraction_type>(Constants::eps)) const = 0;

    virtual std::string to_humanreadable_string() const = 0;
};

inline std::ostream& operator<<(std::ostream& out, const KernelBase& kernel_base) {
    return out << kernel_base.to_humanreadable_string();
}

template <>
struct fmt::formatter<KernelBase> : ostream_formatter { };
