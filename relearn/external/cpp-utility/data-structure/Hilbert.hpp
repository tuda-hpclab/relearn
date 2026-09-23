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

#include "cpp-utility/Exception.hpp"
#include "cpp-utility/data-structure/SpaceFillingCurve.hpp"

#include <cstdint>

namespace utility {

/**
 * @brief A Hilbert curve in 3D: recursively splits the domain into 8 octants per refinement level.
 *      The octants are ordered and oriented so that consecutive one dimensional indices always map to
 *      face-adjacent unit cells, i.e., their coordinates differ by exactly 1 along a single axis.
 *      Unlike Morton, out-of-range input is rejected instead of silently truncated, because the recursive
 *      partitioning (unlike bit-interleaving) has no well-defined meaning for indices or coordinates that
 *      do not belong to the current refinement level.
 */
class Hilbert : public SpaceFillingCurve {
public:
    using coordinates_1d = SpaceFillingCurve::coordinates_1d;
    using coordinates_3d = SpaceFillingCurve::coordinates_3d;

    /**
     * @brief Constructs a new instance of a Hilbert curve with the desired refinement level
     * @param new_refinement_level The desired refinement level
     * @exception Throws an Exception if new_refinement_level > SpaceFillingCurve::max_refinement_level
     */
    constexpr explicit Hilbert(const std::uint8_t new_refinement_level = 0)
        : SpaceFillingCurve(new_refinement_level) {
    }

    constexpr Hilbert(const Hilbert&) = default;
    constexpr Hilbert& operator=(const Hilbert&) = default;
    constexpr Hilbert(Hilbert&&) noexcept = default;
    constexpr Hilbert& operator=(Hilbert&&) noexcept = default;
    // see SpaceFillingCurve.hpp: "= default" here triggers a GCC 11 bug for constexpr virtual destructors
    constexpr ~Hilbert() override { }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain, at the current refinement level
     * @param idx The one dimensional index
     * @exception Throws an Exception if idx >= 2^(3 * get_current_refinement_level())
     * @return The three dimensional index
     */
    [[nodiscard]] constexpr coordinates_3d map_1d_to_3d(const coordinates_1d idx) const override {
        const auto level = get_current_refinement_level();
        const auto number_of_indices = coordinates_1d{ 1 } << (3U * level);

        Exception::check(idx < number_of_indices,
                         "Hilbert::map_1d_to_3d: idx {} is not representable at refinement level {}", idx, level);

        return map_1d_to_3d(idx, level);
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain, at the current refinement level
     * @param coords The three dimensional index
     * @exception Throws an Exception if any component of coords is >= 2^get_current_refinement_level()
     * @return The one dimensional index
     */
    [[nodiscard]] constexpr coordinates_1d map_3d_to_1d(const coordinates_3d& coords) const override {
        const auto level = get_current_refinement_level();
        const auto values_per_axis = coordinates_1d{ 1 } << level;
        const auto& [x, y, z] = coords;

        Exception::check(x < values_per_axis && y < values_per_axis && z < values_per_axis,
                         "Hilbert::map_3d_to_1d: coords {} are not representable at refinement level {}", coords, level);

        return map_3d_to_1d(coords, level);
    }

private:
    [[nodiscard]] constexpr static coordinates_3d map_1d_to_3d(const coordinates_1d idx, const std::uint8_t level) {
        if (level == 0) {
            return coordinates_3d{ 0, 0, 0 };
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
                return coordinates_3d{ 0, 0, 0 };
            }

            if (is_in_back_half && is_in_lower_half && is_in_left_half) {
                return coordinates_3d{ 0, 1, 0 };
            }

            if (is_in_back_half && is_in_upper_half && is_in_left_half) {
                return coordinates_3d{ 0, 1, 1 };
            }

            if (is_in_front_half && is_in_upper_half && is_in_left_half) {
                return coordinates_3d{ 0, 0, 1 };
            }

            if (is_in_front_half && is_in_upper_half && is_in_right_half) {
                return coordinates_3d{ 1, 0, 1 };
            }

            if (is_in_back_half && is_in_upper_half && is_in_right_half) {
                return coordinates_3d{ 1, 1, 1 };
            }

            if (is_in_back_half && is_in_lower_half && is_in_right_half) {
                return coordinates_3d{ 1, 1, 0 };
            }

            if (is_in_front_half && is_in_lower_half && is_in_right_half) {
                return coordinates_3d{ 1, 0, 0 };
            }

            Exception::fail("Hilbert::map_1d_to_3d: Invalid level 1 calculation");
        }

        const auto index_in_octant = idx % number_boxes_eighth;
        const auto coordinates_in_octant = map_1d_to_3d(index_in_octant, static_cast<std::uint8_t>(level - 1));
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
        return coordinates_3d{ y, x, z };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_lower_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ y, z, x };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_upper_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ y, z, x };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_left_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_upper_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - y - 1ULL, z, max_value - x - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_lower_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - y - 1ULL, z, max_value - x - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d translate_right_lower_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - y - 1ULL, max_value - x - 1ULL, z };
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

        const auto index_in_octant = map_3d_to_1d(coordinates_3d{ octant_x, octant_y, octant_z }, static_cast<std::uint8_t>(level - 1));

        auto offset = coordinates_1d{ 0 };

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

        const auto result = index_in_octant + offset;
        return result;
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_lower_front(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ y, x, z };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_lower_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ z, x, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_upper_back(const coordinates_3d& coords, [[maybe_unused]] const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ z, x, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_left_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_upper_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ x, max_value - y - 1ULL, max_value - z - 1ULL };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_upper_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - z - 1ULL, max_value - x - 1ULL, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_lower_back(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - z - 1ULL, max_value - x - 1ULL, y };
    }

    [[nodiscard]] constexpr static coordinates_3d untranslate_right_lower_front(const coordinates_3d& coords, const std::uint64_t max_value) noexcept {
        const auto& [x, y, z] = coords;
        return coordinates_3d{ max_value - y - 1ULL, max_value - x - 1ULL, z };
    }
};

} // namespace utility
