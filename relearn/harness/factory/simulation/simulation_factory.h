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

#include "util/Vec3.h"

#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

class SimulationFactory {
public:
    constexpr static double position_boundary = 10000.0;
    constexpr static std::uint8_t small_refinement_level = 3;
    constexpr static std::uint8_t max_refinement_level = Constants::max_lvl_subdomains;

    static RelearnTypes::bounding_box_type get_random_simulation_box_size(std::mt19937& mt);

    static RelearnTypes::bounding_box_type round_bounding_box(RelearnTypes::bounding_box_type& bb);

    static RelearnTypes::bounding_box_type get_random_small_simulation_box_size(std::mt19937& mt);

    static double get_random_position_element(std::mt19937& mt);

    static double get_random_position_element(std::mt19937& mt, double to_avoid);

    static Vec3d get_random_position(std::mt19937& mt);

    template <typename AllocatorType = std::allocator<Vec3d>>
    static std::vector<Vec3d, AllocatorType> get_random_positions(std::mt19937& mt, RelearnTypes::number_neurons_type num_neurons) {
        auto pos = std::vector<Vec3d, AllocatorType>{};
        pos.reserve(num_neurons);

        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < num_neurons; i++) {
            pos.push_back(get_random_position(mt));
        }

        return pos;
    }

    static Vec3d get_minimum_position();

    static Vec3d get_maximum_position();

    static Vec3d get_random_position_in_box(const Vec3d& min, const Vec3d& max, std::mt19937& mt);

    static Vec3d get_random_position_in_box(const RelearnTypes::bounding_box_type& bb, std::mt19937& mt);

    static std::uint8_t get_random_refinement_level(std::mt19937& mt) noexcept;

    static std::uint8_t get_small_refinement_level(std::mt19937& mt) noexcept;

    static std::uint8_t get_small_positive_refinement_level(std::mt19937& mt) noexcept;

    static std::uint8_t get_large_refinement_level(std::mt19937& mt) noexcept;
};
