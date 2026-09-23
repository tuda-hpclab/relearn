#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/Vec3.h"

#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "factory/random/random_factory.h"

class SimulationFactory {
public:
    constexpr static RelearnTypes::space_type position_boundary = 10000.0;
    constexpr static RelearnTypes::level_type small_refinement_level = 3;
    constexpr static RelearnTypes::level_type max_refinement_level = Constants::max_lvl_subdomains;

    static RelearnTypes::bounding_box_type get_random_simulation_box_size(std::mt19937& mt);

    static RelearnTypes::bounding_box_type round_bounding_box(RelearnTypes::bounding_box_type& bb);

    static RelearnTypes::bounding_box_type get_random_small_simulation_box_size(std::mt19937& mt);

    static RelearnTypes::space_type get_random_position_element(std::mt19937& mt);

    static RelearnTypes::space_type get_random_position_element(std::mt19937& mt, RelearnTypes::space_type to_avoid);

    static RelearnTypes::position_type get_random_position(std::mt19937& mt);

    template <typename AllocatorType = std::allocator<RelearnTypes::position_type>>
    static std::vector<RelearnTypes::position_type, AllocatorType> get_random_positions(std::mt19937& mt, RelearnTypes::number_neurons_type num_neurons) {
        auto pos = std::vector<RelearnTypes::position_type, AllocatorType>{};
        pos.reserve(num_neurons);

        for (auto i = RelearnTypes::number_neurons_type{ 0 }; i < num_neurons; i++) {
            pos.push_back(get_random_position(mt));
        }

        return pos;
    }

    static RelearnTypes::position_type get_minimum_position();

    static RelearnTypes::position_type get_maximum_position();

    static RelearnTypes::position_type get_random_position_in_box(const RelearnTypes::position_type& min, const RelearnTypes::position_type& max, std::mt19937& mt);

    static RelearnTypes::position_type get_random_position_in_box(const RelearnTypes::bounding_box_type& bb, std::mt19937& mt);

    static RelearnTypes::level_type get_random_refinement_level(std::mt19937& mt) noexcept;

    static RelearnTypes::level_type get_small_refinement_level(std::mt19937& mt) noexcept;

    static RelearnTypes::level_type get_small_positive_refinement_level(std::mt19937& mt) noexcept;

    static RelearnTypes::level_type get_large_refinement_level(std::mt19937& mt) noexcept;
};
