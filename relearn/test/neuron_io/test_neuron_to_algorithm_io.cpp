/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_to_algorithm_io.h"

#include "RelearnTest.hpp"

#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "io/NeuronToAlgorithmIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "factory/algorithm/algorithm_config_factory.h"
#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

TEST_F(NeuronToAlgorithmIOTest, testReadAlgorithms) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIRank::root_rank();
    const auto num_algorithms = AlgorithmConfigFactory::get_random_number_algorithms(mt);

    const auto file_path = std::filesystem::path{ "./input_file0.tmp" };

    auto ofstream = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofstream.good();
    const auto is_bad = ofstream.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    auto configs = AlgorithmConfigFactory::create_random_algorithm_configs(num_algorithms, mt);
    auto config_strings = std::vector<std::string>(configs.size());
    std::transform(configs.begin(), configs.end(), config_strings.begin(), [](const AlgorithmConfig& config) {
        return config.to_string();
    });

    for (auto i = 0U; i < config_strings.size(); i++) {
        const auto& config_string = config_strings[i];
        ofstream << config_string << " " << "0:" << (i + 1) << '\n';
    }

    ofstream.close();

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(num_algorithms, mt);
    auto pair = NeuronToAlgorithmIO::read_descriptions(file_path, my_rank, local_group_translator);
    auto read_algorithm_configs = std::move(pair.second);

    for (auto i = 0ULL; i < num_algorithms; ++i) {
        ASSERT_TRUE(read_algorithm_configs[i].is_approximately_equal(configs[i]));
    }

    std::filesystem::remove(file_path);
}

TEST_F(NeuronToAlgorithmIOTest, testIndicesAndNeuronsException1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks. \n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIRank::root_rank();

    const auto num_groups_except_default = 2;

    const auto group_id_to_group_name = NeuronsFactory::get_random_group_names_specific(num_groups_except_default, mt);

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto neuron_id_in_multiple_groups = RandomFactory::get_random_integer<NeuronID::value_type>(0, num_neurons - 1, mt);

    auto neuron_id_to_group_ids = NeuronsFactory::get_random_group_ids({ num_groups_except_default + 1, num_neurons }, mt);

    neuron_id_to_group_ids[neuron_id_in_multiple_groups] = { 0, 1, 2 };

    const auto translator = std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);

    const auto file_path = std::filesystem::path{ "./input_file1.tmp" };

    auto ofstream = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofstream.good();
    const auto is_bad = ofstream.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    ofstream << AlgorithmConfigFactory::create_random_algorithm_config(mt).to_string() << ' ' << translator->get_group_name_for_group_id(1) << '\n';
    ofstream << AlgorithmConfigFactory::create_random_algorithm_config(mt).to_string() << ' ' << translator->get_group_name_for_group_id(2) << '\n';

    ofstream.close();

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronToAlgorithmIO::read_descriptions(file_path, my_rank, translator), RelearnException);

    std::filesystem::remove(file_path);
}

TEST_F(NeuronToAlgorithmIOTest, testIndicesAndNeuronsException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks. \n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIRank::root_rank();

    const auto num_groups_except_default = 1;

    const auto group_id_to_group_name = NeuronsFactory::get_random_group_names_specific(num_groups_except_default, mt);

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto neuron_id_to_group_ids = NeuronsFactory::get_random_group_ids({ num_groups_except_default + 1, num_neurons }, mt, 2);

    const auto translator = std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);

    const auto file_path = std::filesystem::path{ "./input_file2.tmp" };

    auto ofstream = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofstream.good();
    const auto is_bad = ofstream.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    ofstream << AlgorithmConfigFactory::create_random_algorithm_config(mt).to_string() << ' ' << translator->get_group_name_for_group_id(1) << '\n';
    ofstream << AlgorithmConfigFactory::create_random_algorithm_config(mt).to_string() << ' ' << translator->get_group_name_for_group_id(1) << '\n';

    ofstream.close();

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronToAlgorithmIO::read_descriptions(file_path, my_rank, translator), RelearnException);

    std::filesystem::remove(file_path);
}

TEST_F(NeuronToAlgorithmIOTest, testIndicesAndNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks. \n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIRank::root_rank();
    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto num_algorithms = AlgorithmConfigFactory::get_random_number_algorithms(1, std::min(num_neurons, RelearnTypes::number_neurons_type{ 10 }), mt);

    const auto num_groups_except_default = RandomFactory::get_random_integer(num_algorithms, num_neurons, mt); // we have to do this to get at least num_algorithms many group names

    const auto group_names = NeuronsFactory::get_random_group_names_specific(num_groups_except_default, mt);

    const auto neuron_to_group_ids = NeuronsFactory::get_random_group_ids({ num_groups_except_default + 1, num_neurons }, mt, 1, 1); // have neurons in max 1 group (except default), otherwise they would get assigned to multiple configs
    const auto local_group_translator = std::make_shared<LocalGroupTranslator>(group_names, neuron_to_group_ids);

    // this will later determine how many group_names are assigned to a single config
    auto cluster_sizes = std::vector<size_t>(num_algorithms, 1);

    // remaining number of group_names to be assigned
    const auto remaining = num_groups_except_default - num_algorithms;

    // this distributes the number of group_names randomly to the configs
    for (auto i = 0ULL; i < remaining; ++i) {
        const auto index = RandomFactory::get_random_integer<NeuronID::value_type>(NeuronID::value_type{ 0 }, num_algorithms - 1, mt);
        cluster_sizes[index]++;
    }

    auto config_to_groups = std::unordered_map<size_t, RelearnTypes::group_names>{};

    const auto file_path = std::filesystem::path{ "./input_file3.tmp" };

    auto ofstream = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofstream.good();
    const auto is_bad = ofstream.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    auto configs = AlgorithmConfigFactory::create_random_algorithm_configs(num_algorithms, mt);
    auto config_strings = std::vector<std::string>(configs.size());
    std::transform(configs.begin(), configs.end(), config_strings.begin(), [](const AlgorithmConfig& config) {
        return config.to_string();
    });

    size_t group_index = 1; // start with 1 because 0 is the default group which would lead to neurons being assigned to multiple configs
    for (auto i = 0ULL; i < num_algorithms; ++i) {
        ofstream << config_strings[i];

        auto assigned_groups = RelearnTypes::group_names{};
        for (auto j = 0UL; j < cluster_sizes[i]; ++j) {
            ofstream << " " << group_names[group_index];
            assigned_groups.push_back(group_names[group_index++]);
        }

        config_to_groups[i] = std::move(assigned_groups);
        ofstream << "\n";
    }

    ofstream.close();

    auto pair = NeuronToAlgorithmIO::read_descriptions(file_path, my_rank, local_group_translator);

    std::filesystem::remove(file_path);

    const auto& indices_and_neurons = pair.first;

    for (const auto& tuple : indices_and_neurons) {
        const auto& [index, read_neuron_ids] = tuple;

        const auto& group_ids = local_group_translator->translate_group_names_to_group_ids_ordered(config_to_groups.at(index));

        const auto& actual_neuron_ids = local_group_translator->get_neuron_ids_in_groups(group_ids);

        ASSERT_EQ(read_neuron_ids, actual_neuron_ids);
    }

    std::filesystem::remove(file_path);
}