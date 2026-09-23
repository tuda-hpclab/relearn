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

#include "neurons/enums/SynapticElementType.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/Vec3.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

/**
 * This struct contains information needed to create random groups.
 */
struct GroupConfig {
    std::size_t number_groups;
    RelearnTypes::number_neurons_type number_neurons;
};

class NeuronsFactory {
public:
    constexpr static std::size_t max_num_groups_per_neuron_except_default = 5;

    static void generate_random_neurons(std::vector<RelearnTypes::position_type>& positions,
                                        std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids,
                                        RelearnTypes::group_names& group_id_to_group_name,
                                        std::vector<SignalType>& types, std::mt19937& mt,
                                        const NeuronOptFilePaths& paths = NeuronOptFilePaths{});

    static void generate_random_neuron_positions_and_signals(std::vector<RelearnTypes::position_type>& positions,
                                                             std::vector<SignalType>& types,
                                                             std::mt19937& mt, const std::optional<std::filesystem::path>& path = std::nullopt);

    static void generate_random_neuron_groups(std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids,
                                              RelearnTypes::group_names& group_id_to_group_name,
                                              RelearnTypes::number_neurons_type number_neurons,
                                              std::mt19937& mt, const std::optional<std::filesystem::path>& path = std::nullopt,
                                              const std::optional<std::size_t>& _max_groups_except_default = std::nullopt,
                                              std::size_t _max_num_groups_per_neuron_except_default = max_num_groups_per_neuron_except_default);

    static std::vector<RelearnTypes::group_ids> get_random_group_ids(const GroupConfig& group_config, std::mt19937& mt,const std::size_t _min_num_groups_per_neuron_except_default = 0, std::size_t _max_num_groups_per_neuron_except_default = max_num_groups_per_neuron_except_default);

    static RelearnTypes::group_names get_random_group_names(std::size_t max_groups_except_default, std::mt19937& mt);

    static RelearnTypes::group_name get_random_group_name(std::mt19937& mt);

    static RelearnTypes::group_names get_random_group_names_specific(std::size_t number_groups_except_default, std::mt19937& mt);

    static std::vector<std::pair<RelearnTypes::position_type, NeuronID>> generate_random_neurons(const RelearnTypes::position_type& min, const RelearnTypes::position_type& max, size_t max_id, std::mt19937& mt);

    static std::vector<RelearnTypes::group_names> get_neuron_id_vs_group_names(const std::vector<RelearnTypes::group_ids>& neuron_id_vs_group_ids,
                                                                               const RelearnTypes::group_names& group_id_vs_group_name);

    static std::vector<RelearnTypes::group_names_unordered> get_neuron_id_vs_group_names_unordered(const std::vector<RelearnTypes::group_ids_unordered>& neuron_id_to_group_ids_unordered,
                                                                                                   const RelearnTypes::group_names& group_id_to_group_name);

    static std::vector<RelearnTypes::group_ids_unordered> get_neuron_id_to_group_ids_unordered(const std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids);

    static std::string get_invalid_group_name(const RelearnTypes::group_names& group_id_to_group_name, std::mt19937& mt);
};
