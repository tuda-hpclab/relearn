#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/data-structure/SpaceFillingCurve.hpp"

#include <cstdint>

namespace utility {

/**
 * @brief A Morton (Z-order) curve in 3D: bit i of the x, y, and z components is interleaved into bits
 *      3*i, 3*i + 1, and 3*i + 2 of the one dimensional index, respectively.
 */
class Morton : public SpaceFillingCurve {
public:
    using coordinates_1d = SpaceFillingCurve::coordinates_1d;
    using coordinates_3d = SpaceFillingCurve::coordinates_3d;

    /**
     * @brief Constructs a new instance of a Morton curve with the desired refinement level
     * @param new_refinement_level The desired refinement level
     * @exception Throws an Exception if new_refinement_level > SpaceFillingCurve::max_refinement_level
     */
    constexpr explicit Morton(const std::uint8_t new_refinement_level = 0)
        : SpaceFillingCurve(new_refinement_level) {
    }

    constexpr Morton(const Morton&) = default;
    constexpr Morton& operator=(const Morton&) = default;
    constexpr Morton(Morton&&) noexcept = default;
    constexpr Morton& operator=(Morton&&) noexcept = default;
    // see SpaceFillingCurve.hpp: "= default" here triggers a GCC 11 bug for constexpr virtual destructors
    constexpr ~Morton() override { }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain, at the current refinement level
     * @param idx The one dimensional index. Only its lowest 3 * get_current_refinement_level() bits are considered
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr coordinates_3d map_1d_to_3d(const coordinates_1d idx) const override {
        const auto loop_bound = static_cast<std::uint8_t>(get_current_refinement_level() * 3U);

        const auto extract_coordinate = [idx, loop_bound](const std::uint8_t offset) noexcept {
            auto current_value = coordinates_1d{ 0 };
            auto coords_bit = std::uint8_t{ 0 };

            // Takes every third bit from idx (starting at offset) and copies it to current_value
            for (auto idx_bit = offset; idx_bit < loop_bound; idx_bit += 3) {
                current_value = copy_bit(idx, idx_bit, current_value, coords_bit);
                ++coords_bit;
            }

            return current_value;
        };

        // idx has the structure: ... z2 y2 x2 z1 y1 x1 z0 y0 x0
        const auto x_value = extract_coordinate(0);
        const auto y_value = extract_coordinate(1);
        const auto z_value = extract_coordinate(2);

        return coordinates_3d{ x_value, y_value, z_value };
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain, at the current refinement level
     * @param coords The three dimensional index. Only the lowest get_current_refinement_level() bits of every
     *      component are considered
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const override {
        auto result = coordinates_1d{ 0 };
        const auto& [x, y, z] = coords;

        const auto current_refinement_level = get_current_refinement_level();
        for (auto i = std::uint8_t{ 0 }; i < current_refinement_level; ++i) {
            const auto x_bit = select_bit(x, i);
            const auto y_bit = select_bit(y, i);
            const auto z_bit = select_bit(z, i);

            const auto block = (z_bit << 2U) + (y_bit << 1U) + x_bit;
            result |= block << (3U * i);
        }

        return result;
    }

private:
    [[nodiscard]] constexpr static coordinates_1d set_bit(const coordinates_1d variable, const std::uint8_t bit) noexcept {
        return variable | (coordinates_1d{ 1 } << bit);
    }

    [[nodiscard]] constexpr static coordinates_1d unset_bit(const coordinates_1d variable, const std::uint8_t bit) noexcept {
        return variable & ~(coordinates_1d{ 1 } << bit);
    }

    [[nodiscard]] constexpr static coordinates_1d select_bit(const coordinates_1d number, const std::uint8_t bit) noexcept {
        return (number >> bit) & coordinates_1d{ 1 };
    }

    [[nodiscard]] constexpr static coordinates_1d copy_bit(const coordinates_1d source, const std::uint8_t source_bit, const coordinates_1d destination, const std::uint8_t destination_bit) noexcept {
        return select_bit(source, source_bit) != 0 ? set_bit(destination, destination_bit) : unset_bit(destination, destination_bit);
    }
};

} // namespace utility
