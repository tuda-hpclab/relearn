/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "simulation_factory.h"

#include "types/SpaceTypes.h"
#include "util/Vec3.h"

#include "factory/random/random_factory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

RelearnTypes::bounding_box_type SimulationFactory::get_random_simulation_box_size(std::mt19937& mt) {
    const auto rand_x_1 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    const auto rand_x_2 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);

    const auto rand_y_1 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    const auto rand_y_2 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);

    const auto rand_z_1 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    const auto rand_z_2 = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);

    return {
        { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
        { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
    };
}

RelearnTypes::bounding_box_type SimulationFactory::round_bounding_box(RelearnTypes::bounding_box_type& bb) {
    const auto& [min, max] = bb;
    const auto& [x_min, y_min, z_min] = min;
    const auto& [x_max, y_max, z_max] = max;

    const auto x_min_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(x_min));
    const auto y_min_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(y_min));
    const auto z_min_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(z_min));

    const auto x_max_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(x_max));
    const auto y_max_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(y_max));
    const auto z_max_rounded = static_cast<RelearnTypes::space_type>(static_cast<std::int64_t>(z_max));

    const auto min_rounded = Vec3{ x_min_rounded, y_min_rounded, z_min_rounded };
    const auto max_rounded = Vec3{ x_max_rounded, y_max_rounded, z_max_rounded };

    return { min_rounded, max_rounded };
}

RelearnTypes::bounding_box_type SimulationFactory::get_random_small_simulation_box_size(std::mt19937& mt) {
    const auto rand_x_1 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);
    const auto rand_x_2 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);

    const auto rand_y_1 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);
    const auto rand_y_2 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);

    const auto rand_z_1 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);
    const auto rand_z_2 = RandomFactory::get_random_double(-position_boundary / RelearnTypes::space_type{ 10 }, +position_boundary / RelearnTypes::space_type{ 10 }, mt);

    return {
        { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
        { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
    };
}

RelearnTypes::space_type SimulationFactory::get_random_position_element(std::mt19937& mt) {
    const auto val = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    return val;
}

RelearnTypes::space_type SimulationFactory::get_random_position_element(std::mt19937& mt, const RelearnTypes::space_type to_avoid) {
    while (true) {
        const auto val = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
        if (val != to_avoid) {
            return static_cast<RelearnTypes::space_type>(val);
        }
    }
}

RelearnTypes::position_type SimulationFactory::get_random_position(std::mt19937& mt) {
    const auto x = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    const auto y = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);
    const auto z = RandomFactory::get_random_double(-position_boundary, +position_boundary, mt);

    return { x, y, z };
}

RelearnTypes::position_type SimulationFactory::get_minimum_position() {
    return { -position_boundary, -position_boundary, -position_boundary };
}

RelearnTypes::position_type SimulationFactory::get_maximum_position() {
    return { position_boundary, position_boundary, position_boundary };
}

RelearnTypes::position_type SimulationFactory::get_random_position_in_box(const RelearnTypes::position_type& min, const RelearnTypes::position_type& max, std::mt19937& mt) {
    const auto x = RandomFactory::get_random_double(min.get_x(), max.get_x(), mt);
    const auto y = RandomFactory::get_random_double(min.get_y(), max.get_y(), mt);
    const auto z = RandomFactory::get_random_double(min.get_z(), max.get_z(), mt);

    return { x, y, z };
}

RelearnTypes::position_type SimulationFactory::get_random_position_in_box(const RelearnTypes::bounding_box_type& bb, std::mt19937& mt) {
    const auto& [min, max] = bb;
    return get_random_position_in_box(min, max, mt);
}

RelearnTypes::level_type SimulationFactory::get_random_refinement_level(std::mt19937& mt) noexcept {
    return static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::size_t>(0, max_refinement_level, mt));
}

RelearnTypes::level_type SimulationFactory::get_small_refinement_level(std::mt19937& mt) noexcept {
    return static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::size_t>(0, small_refinement_level, mt));
}

RelearnTypes::level_type SimulationFactory::get_small_positive_refinement_level(std::mt19937& mt) noexcept {
    return static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::size_t>(1, small_refinement_level, mt));
}

RelearnTypes::level_type SimulationFactory::get_large_refinement_level(std::mt19937& mt) noexcept {
    return static_cast<RelearnTypes::level_type>(RandomFactory::get_random_integer<std::size_t>(small_refinement_level + 1, max_refinement_level, mt));
}
