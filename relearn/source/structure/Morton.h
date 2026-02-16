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

#include "structure/SpaceFillingCurve.h"

#include <cstdint>

/**
 * This class represents a MortonCurve in 3D.
 */
class Morton : public SpaceFillingCurve {
public:
    using coordinates_1d = SpaceFillingCurve::coordinates_1d;
    using coordinates_3d = SpaceFillingCurve::coordinates_3d;

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param _refinement_level  The desired refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    constexpr explicit Morton(const std::uint8_t _refinement_level = 0)
        : SpaceFillingCurve(_refinement_level) { }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr coordinates_3d map_1d_to_3d(const coordinates_1d idx) const override {
        const auto extract_coordinate = [idx](const std::uint8_t offset) noexcept {
            constexpr auto loop_bound = Constants::max_lvl_subdomains * 3;

            auto current_value = std::uint64_t{ 0 };
            auto coords_bit = std::uint8_t{ 0 };

            // Takes every third bit from idx (starting at offset) and copies it to current_value
            for (auto idx_bit = offset; idx_bit < loop_bound; idx_bit += 3) {
                const auto old = current_value;
                const auto new_val = copy_bit(idx, idx_bit, old, coords_bit);
                current_value = new_val;
                ++coords_bit;
            }

            return current_value;
        };

        // The index has structure: ...... z2 y2 x2 z1 y1 x1 z0 y0 x0
        const auto x_value = extract_coordinate(0);
        const auto y_value = extract_coordinate(1);
        const auto z_value = extract_coordinate(2);

        return { x_value, y_value, z_value };
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param coords The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const override {
        auto result = std::uint64_t{ 0 };
        const auto _refinement_level = get_current_refinement_level();

        for (auto i = std::uint8_t{ 0 }; i < _refinement_level; ++i) {
            const auto& [x, y, z] = coords;

            const auto x_bit = select_bit(x, i);
            const auto y_bit = select_bit(y, i);
            const auto z_bit = select_bit(z, i);

            const auto block = (z_bit << 2U) + (y_bit << 1U) + x_bit;
            result |= block << (3U * i);
        }

        return result;
    }

private:
    [[nodiscard]] constexpr static std::uint64_t set_bit(const std::uint64_t variable, const std::uint8_t bit) noexcept {
        const auto val = variable | (static_cast<std::uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] constexpr static std::uint64_t unset_bit(const std::uint64_t variable, const std::uint8_t bit) noexcept {
        const auto val = variable & ~(static_cast<std::uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] constexpr static std::uint64_t select_bit(const std::uint64_t number, const std::uint8_t bit) noexcept {
        return ((number >> bit) & 1U);
    }

    [[nodiscard]] constexpr static std::uint64_t copy_bit(const std::uint64_t source, const std::uint8_t source_bit, const std::uint64_t destination, const std::uint8_t destination_bit) noexcept {
        // A simpler solution might be:
        // destination ^= (-select_bit(source, source_bit) ^ destination) & (1 << destination_bit);

        const auto bit_in_source = select_bit(source, source_bit);
        if (1 == bit_in_source) {
            const auto return_value = set_bit(destination, destination_bit);
            return return_value;
        }

        const auto return_value = unset_bit(destination, destination_bit);
        return return_value;
    }
};
