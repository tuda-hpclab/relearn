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
 * This class represents a HilbertCurve in 3D.
 */
class Hilbert : public SpaceFillingCurve {
public:
    using coordinates_1d = SpaceFillingCurve::coordinates_1d;
    using coordinates_3d = SpaceFillingCurve::coordinates_3d;

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param _refinement_level  The desired refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    constexpr explicit Hilbert(const std::uint8_t _refinement_level = 0)
        : SpaceFillingCurve(_refinement_level) { }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr coordinates_3d map_1d_to_3d(const coordinates_1d idx) const override {
        const auto _refinement_level = get_current_refinement_level();
        return map_1d_to_3d(idx, _refinement_level);
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param coords The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const override {
        const auto _refinement_level = get_current_refinement_level();
        return map_3d_to_1d(coords, _refinement_level);
    }

private:
    [[nodiscard]] constexpr static coordinates_3d map_1d_to_3d(const coordinates_1d idx, const std::uint8_t level) {
        if (level == 0) {
            return { 0, 0, 0 };
        }

        const auto number_boxes_per_dimension = std::uint64_t{ 1 } << (level - 1ULL);
        const auto number_boxes = std::uint64_t{ 1 } << (level * 3ULL);
        const auto number_boxes_half = number_boxes >> 1ULL;
        const auto number_boxes_quarter = number_boxes >> 2ULL;
        const auto number_boxes_eighth = number_boxes >> 3ULL;

        const auto is_in_left_half = idx < number_boxes_half;
        const auto is_in_right_half = !is_in_left_half;

        const auto is_in_lower_half = (idx < number_boxes_quarter) || (idx >= number_boxes - number_boxes_quarter);
        const auto is_in_upper_half = !is_in_lower_half;

        const auto is_in_front_half = (idx < number_boxes_eighth) || ((idx >= number_boxes_half - number_boxes_eighth) && (idx < number_boxes_half + number_boxes_eighth)) || (idx >= number_boxes - number_boxes_eighth);
        const auto is_in_back_half = !is_in_front_half;

        if (level == 1) {
            if (is_in_front_half && is_in_lower_half && is_in_left_half) {
                return { 0, 0, 0 };
            }

            if (is_in_back_half && is_in_lower_half && is_in_left_half) {
                return { 0, 1, 0 };
            }

            if (is_in_back_half && is_in_upper_half && is_in_left_half) {
                return { 0, 1, 1 };
            }

            if (is_in_front_half && is_in_upper_half && is_in_left_half) {
                return { 0, 0, 1 };
            }

            if (is_in_front_half && is_in_upper_half && is_in_right_half) {
                return { 1, 0, 1 };
            }

            if (is_in_back_half && is_in_upper_half && is_in_right_half) {
                return { 1, 1, 1 };
            }

            if (is_in_back_half && is_in_lower_half && is_in_right_half) {
                return { 1, 1, 0 };
            }

            if (is_in_front_half && is_in_lower_half && is_in_right_half) {
                return { 1, 0, 0 };
            }

            RelearnException::fail("Hilbert::map_1d_to_3d: Invalid level 1 calculation");
        }

        const auto index_in_octant = idx % number_boxes_eighth;
        const auto coordinates_in_octant = map_1d_to_3d(index_in_octant, level - 1);
        auto [octant_x, octant_y, octant_z] = coordinates_in_octant;

        if (is_in_front_half && is_in_lower_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = translate_left_lower_front(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_lower_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = translate_left_lower_back(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_upper_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = translate_left_upper_back(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_upper_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = translate_left_upper_front(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_upper_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = translate_right_upper_front(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_upper_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = translate_right_upper_back(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_lower_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = translate_right_lower_back(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_lower_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = translate_right_lower_front(coordinates_in_octant, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        auto offset_vector = coordinates_3d{ 0ULL, 0ULL, 0ULL };

        if (is_in_front_half && is_in_lower_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, 0ULL, 0ULL };
        }

        if (is_in_back_half && is_in_lower_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, number_boxes_per_dimension, 0ULL };
        }

        if (is_in_back_half && is_in_upper_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, number_boxes_per_dimension, number_boxes_per_dimension };
        }

        if (is_in_front_half && is_in_upper_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, 0ULL, number_boxes_per_dimension };
        }

        if (is_in_front_half && is_in_upper_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, 0ULL, number_boxes_per_dimension };
        }

        if (is_in_back_half && is_in_upper_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, number_boxes_per_dimension, number_boxes_per_dimension };
        }

        if (is_in_back_half && is_in_lower_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, number_boxes_per_dimension, 0ULL };
        }

        if (is_in_front_half && is_in_lower_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, 0ULL, 0ULL };
        }

        const auto octant_vector = coordinates_3d{ octant_x, octant_y, octant_z };

        const auto result = octant_vector + offset_vector;
        return result;
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_lower_front(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { y, x, z };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_lower_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { y, z, x };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_upper_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { y, z, x };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_upper_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - y - 1ULL, z, max_value - x - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_lower_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - y - 1ULL, z, max_value - x - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_lower_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - y - 1ULL, max_value - x - 1ULL, z };
    }

    [[nodiscard]] constexpr static coordinates_1d map_3d_to_1d(const coordinates_3d& coords, const std::uint8_t level) noexcept {
        if (level == 0) {
            return 0;
        }

        const auto& [x, y, z] = coords;

        if (level == 1) {
            if (x == 0 && y == 0 && z == 0) {
                return 0;
            }

            if (x == 0 && y == 1 && z == 0) {
                return 1;
            }

            if (x == 0 && y == 1 && z == 1) {
                return 2;
            }

            if (x == 0 && y == 0 && z == 1) {
                return 3;
            }

            if (x == 1 && y == 0 && z == 1) {
                return 4;
            }

            if (x == 1 && y == 1 && z == 1) {
                return 5;
            }

            if (x == 1 && y == 1 && z == 0) {
                return 6;
            }

            if (x == 1 && y == 0 && z == 0) {
                return 7;
            }
        }

        const auto number_boxes_per_dimension = std::uint64_t{ 1 } << (level - 1ULL);
        const auto number_boxes = std::uint64_t{ 1 } << (level * 3ULL);
        const auto number_boxes_half = number_boxes >> 1ULL;
        const auto number_boxes_quarter = number_boxes >> 2ULL;
        const auto number_boxes_eighth = number_boxes >> 3ULL;

        const auto is_in_left_half = x < number_boxes_per_dimension;
        const auto is_in_right_half = !is_in_left_half;

        const auto is_in_lower_half = z < number_boxes_per_dimension;
        const auto is_in_upper_half = !is_in_lower_half;

        const auto is_in_front_half = y < number_boxes_per_dimension;
        const auto is_in_back_half = !is_in_front_half;

        auto offset_vector = coordinates_3d{ 0ULL, 0ULL, 0ULL };

        if (is_in_front_half && is_in_lower_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, 0ULL, 0ULL };
        }

        if (is_in_back_half && is_in_lower_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, number_boxes_per_dimension, 0ULL };
        }

        if (is_in_back_half && is_in_upper_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, number_boxes_per_dimension, number_boxes_per_dimension };
        }

        if (is_in_front_half && is_in_upper_half && is_in_left_half) {
            offset_vector = coordinates_3d{ 0ULL, 0ULL, number_boxes_per_dimension };
        }

        if (is_in_front_half && is_in_upper_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, 0ULL, number_boxes_per_dimension };
        }

        if (is_in_back_half && is_in_upper_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, number_boxes_per_dimension, number_boxes_per_dimension };
        }

        if (is_in_back_half && is_in_lower_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, number_boxes_per_dimension, 0ULL };
        }

        if (is_in_front_half && is_in_lower_half && is_in_right_half) {
            offset_vector = coordinates_3d{ number_boxes_per_dimension, 0ULL, 0ULL };
        }

        const auto octant_vector = coords - offset_vector;
        auto [octant_x, octant_y, octant_z] = octant_vector;

        if (is_in_front_half && is_in_lower_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = untranslate_left_lower_front(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_lower_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = untranslate_left_lower_back(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_upper_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = untranslate_left_upper_back(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_upper_half && is_in_left_half) {
            const auto translated_coordinated_in_octant = untranslate_left_upper_front(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_upper_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = untranslate_right_upper_front(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_upper_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = untranslate_right_upper_back(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_back_half && is_in_lower_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = untranslate_right_lower_back(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        if (is_in_front_half && is_in_lower_half && is_in_right_half) {
            const auto translated_coordinated_in_octant = untranslate_right_lower_front(octant_vector, number_boxes_per_dimension);
            const auto& [translated_x, translated_y, translated_z] = translated_coordinated_in_octant;

            octant_x = translated_x;
            octant_y = translated_y;
            octant_z = translated_z;
        }

        const auto index_in_octant = map_3d_to_1d({ octant_x, octant_y, octant_z }, level - 1);

        auto offset = 0ULL;

        if (is_in_front_half && is_in_lower_half && is_in_left_half) {
            offset = 0ULL;
        }

        if (is_in_back_half && is_in_lower_half && is_in_left_half) {
            offset = number_boxes_eighth;
        }

        if (is_in_back_half && is_in_upper_half && is_in_left_half) {
            offset = number_boxes_quarter;
        }

        if (is_in_front_half && is_in_upper_half && is_in_left_half) {
            offset = number_boxes_quarter + number_boxes_eighth;
        }

        if (is_in_front_half && is_in_upper_half && is_in_right_half) {
            offset = number_boxes_half;
        }

        if (is_in_back_half && is_in_upper_half && is_in_right_half) {
            offset = number_boxes_half + number_boxes_eighth;
        }

        if (is_in_back_half && is_in_lower_half && is_in_right_half) {
            offset = number_boxes_half + number_boxes_quarter;
        }

        if (is_in_front_half && is_in_lower_half && is_in_right_half) {
            offset = number_boxes_half + number_boxes_quarter + number_boxes_eighth;
        }

        auto result = index_in_octant + offset;
        return result;
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_lower_front(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { y, x, z };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_lower_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { z, x, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_upper_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { z, x, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_upper_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - z - 1ULL, max_value - x - 1ULL, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_lower_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - z - 1ULL, max_value - x - 1ULL, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_lower_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return { max_value - y - 1ULL, max_value - x - 1ULL, z };
    }
};
