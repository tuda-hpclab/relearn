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

#include "Types.h"

#include <optional>

/**
 * This type is used to represent a virtual plasticity element,
 * i.e., axons and dendrites, when it comes to combining multiple of them
 * in the octree. Does not use std::optional
 */
class VirtualPlasticityElementManual {
public:
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;

    /**
     * @brief Sets the number of free elements
     * @param number_free_elements The number of free elements
     */
    constexpr void set_number_free_elements(const counter_type number_free_elements) noexcept {
        num_free_elements = number_free_elements;
    }

    /**
     * @brief Returns the number of free elements
     * @return The number of free elements
     */
    [[nodiscard]] constexpr counter_type get_number_free_elements() const noexcept {
        return num_free_elements;
    }

    /**
     * @brief Sets the position of this plasticity element. Can be empty
     * @param virtual_position The new position
     */
    constexpr void set_position(const std::optional<position_type>& virtual_position) noexcept {
        const auto valid_pos = virtual_position.has_value();
        is_valid = valid_pos;

        if (valid_pos) {
            position = virtual_position.value();
        }
    }

    /**
     * @brief Returns the position of this plasticity element. Can be empty
     * @return The current position
     */
    [[nodiscard]] constexpr std::optional<position_type> get_position() const noexcept {
        if (!is_valid) {
            return {};
        }

        return position;
    }

    /**
     * @brief Compare two VirtualPlasticityElementOptional
     * @return True iff both are considered equal;
     *      which is the case if both have no position, or both have the same position and number of free elements
     */
    [[nodiscard]] friend constexpr bool operator==(const VirtualPlasticityElementManual& first, const VirtualPlasticityElementManual& second) noexcept {
        const auto first_has_value = first.is_valid;
        const auto second_has_value = second.is_valid;

        if (first_has_value != second_has_value) {
            return false;
        }

        // second_has_value == first_has_value
        if (!first_has_value) {
            return true;
        }

        return first.position == second.position && first.num_free_elements == second.num_free_elements;
    }

private:
    // Avoiding std::optional<> saves 8 bytes, which translates to 32 bytes per FFM-cell

    position_type position{};
    counter_type num_free_elements{ 0 };
    bool is_valid{ false };
};

/**
 * This type is used to represent a virtual plasticity element,
 * i.e., axons and dendrites, when it comes to combining multiple of them
 * in the octree. Uses std::optional
 */
class VirtualPlasticityElementOptional {
public:
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;

    /**
     * @brief Sets the number of free elements
     * @param number_free_elements The number of free elements
     */
    constexpr void set_number_free_elements(const counter_type number_free_elements) noexcept {
        num_free_elements = number_free_elements;
    }

    /**
     * @brief Returns the number of free elements
     * @return The number of free elements
     */
    [[nodiscard]] constexpr counter_type get_number_free_elements() const noexcept {
        return num_free_elements;
    }

    /**
     * @brief Sets the position of this plasticity element. Can be empty
     * @param virtual_position The new position
     */
    constexpr void set_position(const std::optional<position_type>& virtual_position) noexcept {
        position = virtual_position;
    }

    /**
     * @brief Returns the position of this plasticity element. Can be empty
     * @return The current position
     */
    [[nodiscard]] constexpr std::optional<position_type> get_position() const noexcept {
        return position;
    }

    /**
     * @brief Compare two VirtualPlasticityElementOptional
     * @return True iff both are considered equal;
     *      which is the case if both have no position, or both have the same position and number of free elements
     */
    [[nodiscard]] friend constexpr bool operator==(const VirtualPlasticityElementOptional& first, const VirtualPlasticityElementOptional& second) noexcept {
        const auto first_has_value = first.position.has_value();
        const auto second_has_value = second.position.has_value();

        if (first_has_value != second_has_value) {
            return false;
        }

        // second_has_value == first_has_value
        if (!first_has_value) {
            return true;
        }

        return first.position.value() == second.position.value() && first.num_free_elements == second.num_free_elements;
    }

private:
    std::optional<position_type> position{};
    counter_type num_free_elements{ 0 };
};

// Switch for the implementation of VPE (space vs. time trade-off)
using VirtualPlasticityElement = VirtualPlasticityElementManual;
