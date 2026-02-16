/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_local_group_translator.h"

#include "Config.h"
#include "RelearnTest.hpp"
#include "Types.h"

#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

TEST_F(LocalGroupTranslatorTest, testEssentials) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto num_groups_max = std::min(NeuronID::value_type{ 50 }, num_neurons);
    auto group_id_to_group_name = RelearnTypes::group_names{};
    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, num_neurons, mt, std::nullopt, num_groups_max);
    const auto cp_group_id_to_group_name = group_id_to_group_name;
    const auto cp_neuron_id_to_group_ids = neuron_id_to_group_ids;

    const auto translator = LocalGroupTranslator(group_id_to_group_name, neuron_id_to_group_ids);

    ASSERT_EQ(group_id_to_group_name.size(), translator.get_number_of_groups());
    ASSERT_EQ(num_neurons, translator.get_number_neurons_in_total());

    for (const auto neuron_id : NeuronID::range_id(num_neurons)) {
        ASSERT_EQ(cp_neuron_id_to_group_ids[neuron_id], translator.get_group_ids_for_neuron_id(neuron_id));

        const auto& group_ids = cp_neuron_id_to_group_ids[neuron_id];
        const auto group_names = group_ids
                                 | ranges::views::transform(utility::lookup(group_id_to_group_name))
                                 | ranges::to_vector;
        ASSERT_EQ(group_names, translator.get_group_names_for_neuron_id(neuron_id));
        ASSERT_EQ(group_names, translator.get_group_names_for_group_ids(group_ids));
        ASSERT_EQ(group_ids, translator.translate_group_names_to_group_ids_ordered(group_names));

        const auto& group_ids_unordered = translator.get_group_ids_for_neuron_id_unordered(neuron_id);
        for (const auto& group_id : group_ids) {
            ASSERT_TRUE(group_ids_unordered.contains(group_id));
        }

        const auto& group_names_unordered = translator.get_group_names_for_neuron_id_unordered(neuron_id);
        for (const auto& group_name : group_names) {
            ASSERT_TRUE(group_names_unordered.contains(group_name));
            ASSERT_TRUE(translator.knows_group_name(group_name));
        }
        ASSERT_EQ(group_names_unordered, translator.get_group_names_for_group_ids_unordered(group_ids_unordered));
        ASSERT_EQ(group_ids_unordered, translator.translate_group_names_to_group_ids(group_names_unordered));
    }

    const auto random_invalid_group_name = NeuronsFactory::get_invalid_group_name(group_id_to_group_name, mt);
    ASSERT_FALSE(translator.knows_group_name(random_invalid_group_name));

    for (const auto group_id : ranges::views::indices(group_id_to_group_name.size())) {
        ASSERT_EQ(cp_group_id_to_group_name[group_id], translator.get_group_name_for_group_id(group_id));
        ASSERT_EQ(translator.get_group_id_for_group_name(cp_group_id_to_group_name[group_id]), group_id);
    }
}

TEST_F(LocalGroupTranslatorTest, testNumberNeuronsConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto translator = LocalGroupTranslator(number_neurons);

    const auto& neuron_id_to_group_ids = translator.get_neuron_ids_to_group_ids();
    const auto& neuron_id_to_group_ids_unordered = translator.get_neuron_ids_to_group_ids_unordered();
    const auto& group_id_to_group_name = translator.get_all_group_names();

    ASSERT_EQ(group_id_to_group_name, RelearnTypes::group_names{ std::string{ Constants::default_group_name } });

    for (auto neuron_id = 0UL; neuron_id < neuron_id_to_group_ids.size(); ++neuron_id) {
        const auto& group_ids = neuron_id_to_group_ids[neuron_id];
        const auto& group_ids_unordered = neuron_id_to_group_ids_unordered[neuron_id];

        ASSERT_EQ(group_ids, RelearnTypes::group_ids{ Constants::default_group_id });
        ASSERT_EQ(group_ids_unordered, RelearnTypes::group_ids_unordered{ Constants::default_group_id });
    }
}

TEST_F(LocalGroupTranslatorTest, testFileConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto lgt_tmp = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    const auto number_neurons = lgt_tmp->get_number_neurons_in_total();

    const auto path = std::filesystem::path{ "./groups.tmp" };

    NeuronIO::write_neuron_groups(path, lgt_tmp);

    const auto translator = std::make_shared<LocalGroupTranslator>(path, number_neurons);

    const auto& golden_neuron_id_to_group_ids_unordered = lgt_tmp->get_neuron_ids_to_group_ids_unordered();
    const auto& golden_group_id_to_group_name = lgt_tmp->get_all_group_names();

    const auto& neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();
    const auto& group_id_to_group_name = translator->get_all_group_names();

    ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, neuron_id_to_group_ids_unordered); // only test with unordered group ids because order of group ids in regular group ids can be different
    ASSERT_EQ(golden_group_id_to_group_name, group_id_to_group_name);

    std::filesystem::remove(path);
}

TEST_F(LocalGroupTranslatorTest, testConstructorExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    const auto group_id_to_group_name = NeuronsFactory::get_random_group_names_specific(num_neurons, mt);
    const auto neuron_id_to_group_ids = NeuronsFactory::get_random_group_ids({ group_id_to_group_name.size(), num_neurons }, mt);

    auto one_wrong_group_id = neuron_id_to_group_ids;
    const auto ind1 = RandomFactory::get_random_integer(std::size_t{ 0 }, one_wrong_group_id.size() - 1, mt);
    one_wrong_group_id[ind1] = { num_neurons + 1 };

    auto duplicated_group_name = group_id_to_group_name;
    const auto ind2 = RandomFactory::get_random_integer(std::size_t{ 0 }, duplicated_group_name.size() - 1, mt);
    const auto ind3 = RandomFactory::get_random_integer(std::size_t{ 0 }, duplicated_group_name.size() - 1, ind2, mt);
    duplicated_group_name[ind2] = duplicated_group_name[ind3];

    auto neuron_has_no_groups = neuron_id_to_group_ids;
    const auto ind4 = RandomFactory::get_random_integer(std::size_t{ 0 }, neuron_has_no_groups.size() - 1, mt);
    neuron_has_no_groups[ind4].clear();

    // first constructor
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(duplicated_group_name, neuron_id_to_group_ids), RelearnException);
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(group_id_to_group_name, one_wrong_group_id), RelearnException);
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(group_id_to_group_name, neuron_has_no_groups), RelearnException);

    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(RelearnTypes::group_names({}), neuron_id_to_group_ids), RelearnException);
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(RelearnTypes::group_names({}), std::vector<RelearnTypes::group_ids>({})), RelearnException);
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(group_id_to_group_name, std::vector<RelearnTypes::group_ids>({})), RelearnException);

    // second constructor
    ASSERT_THROW_NO_PRINT(LocalGroupTranslator(0), RelearnException);

    const auto translator = std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);
    ASSERT_EQ(num_neurons + 1, translator->get_number_of_groups());
    ASSERT_EQ(num_neurons, translator->get_number_neurons_in_total());
}

TEST_F(LocalGroupTranslatorTest, testGroupGetters) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>(num_neurons);
    auto group0 = std::vector<RelearnTypes::neuron_id>{};
    auto group1 = std::vector<RelearnTypes::neuron_id>{};
    for (const auto neuron_id : NeuronID::range_id(num_neurons)) {
        neuron_id_to_group_ids[neuron_id].emplace_back(0);
        group0.emplace_back(neuron_id);

        if (RandomFactory::get_random_bool(mt)) {
            neuron_id_to_group_ids[neuron_id].emplace_back(1);
            group1.emplace_back(neuron_id);
        }
    }

    const auto group_id_to_group_name = NeuronsFactory::get_random_group_names_specific(2, mt);
    const auto translator = LocalGroupTranslator(group_id_to_group_name, neuron_id_to_group_ids);

    ASSERT_EQ(group0.size(), translator.get_number_neurons_in_group(0));
    ASSERT_EQ(group1.size(), translator.get_number_neurons_in_group(1));

    const auto read_group0 = translator.get_neuron_ids_in_group(0);
    for (const auto& neuron_id : read_group0) {
        ASSERT_TRUE(ranges::contains(group0, neuron_id.get_neuron_id()));
    }

    const auto read_group1 = translator.get_neuron_ids_in_group(1);
    for (const auto& neuron_id : read_group1) {
        ASSERT_TRUE(ranges::contains(group1, neuron_id.get_neuron_id()));
    }

    const auto read2_group0 = translator.get_neuron_ids_in_groups(RelearnTypes::group_ids{ 0 });
    for (const auto& neuron_id : read2_group0) {
        ASSERT_TRUE(ranges::contains(group0, neuron_id.get_neuron_id()));
    }

    const auto read2_group1 = translator.get_neuron_ids_in_groups(RelearnTypes::group_ids{ 1 });
    for (const auto& neuron_id : read2_group1) {
        ASSERT_TRUE(ranges::contains(group1, neuron_id.get_neuron_id()));
    }

    const auto read3_group0 = translator.get_neuron_ids_in_groups(RelearnTypes::group_ids_unordered{ 0 });
    for (const auto& neuron_id : read3_group0) {
        ASSERT_TRUE(ranges::contains(group0, neuron_id.get_neuron_id()));
    }

    const auto read3_group1 = translator.get_neuron_ids_in_groups(RelearnTypes::group_ids_unordered{ 1 });
    for (const auto& neuron_id : read3_group1) {
        ASSERT_TRUE(ranges::contains(group1, neuron_id.get_neuron_id()));
    }

    const auto read_all = translator.get_neuron_ids_in_groups(RelearnTypes::group_ids{ 0, 1 });
    ASSERT_TRUE(read_all.size() <= num_neurons);
}

TEST_F(LocalGroupTranslatorTest, testGetterExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;

    auto group_id_to_group_name = NeuronsFactory::get_random_group_names_specific(RandomFactory::get_random_integer(NeuronID::value_type{ 1 }, num_neurons, mt), mt);
    const auto num_groups = group_id_to_group_name.size();
    const auto neuron_id_to_group_ids = NeuronsFactory::get_random_group_ids({ group_id_to_group_name.size(), num_neurons }, mt);

    const auto translator = LocalGroupTranslator(group_id_to_group_name, neuron_id_to_group_ids);

    ASSERT_THROW_NO_PRINT(std::ignore = translator.get_group_names_for_neuron_id(num_neurons), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = translator.get_group_ids_for_neuron_id(num_neurons), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = translator.get_group_name_for_group_id(num_groups), RelearnException);

    const auto percentage = RandomFactory::get_random_percentage<double>(mt);

    ASSERT_THROW_NO_PRINT(std::ignore = translator.get_group_id_for_group_name(std::to_string(percentage)), RelearnException);
}

TEST_F(LocalGroupTranslatorTest, testRegex) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr static auto possible_group_names = std::array{
        "REGFSAIJEX",
        "REGf+EX",
        "REGF+EX",
        "REGuiwnzrsx9EX",
        "REG21sdf65as1dcawEX",
        "REG54asdfasf23EX",
        "REGi7z23i+EX",
        "REG1o24zfgnsclEX",
        "REGasdflajl12EX",
        "REGi1u2h4091231fEX",
        "REGi1u2H4091231fEX",
        "REG1i3h1ih1aasdafasadfEX",
        "REGEXEEXEXEXEXEXEX",
        "REGEXEEXEXEXEXEX",
        "REGEXEEXEXEXEXEXEXEX",
        "REGjsdfslfjlsjfksdfEX",
        "REGfppppppppppEX",
        "REG971820312kfskfEX",
        "REG0123456789EX",
        "REGppppsafdsfEX",
    };

    auto group_id_to_group_name = RelearnTypes::group_names{ "not_regex", "not_regex_too", "definitely_not_regex" };
    group_id_to_group_name.insert(group_id_to_group_name.end(), possible_group_names.begin(), possible_group_names.end());

    const auto neuron_id_to_group_ids = NeuronsFactory::get_random_group_ids({ group_id_to_group_name.size(), 10 }, mt);

    const auto translator = LocalGroupTranslator(group_id_to_group_name, neuron_id_to_group_ids);

    const auto read_group_names = translator.get_matching_group_names("REG.+EX");
    const auto read_group_ids = translator.get_group_ids_for_matching_group_names("REG.+EX");

    for (const auto& group_name : possible_group_names) {
        ASSERT_TRUE(read_group_names.contains(group_name));
        ASSERT_TRUE(read_group_ids.contains(translator.get_group_id_for_group_name(group_name)));
    }
    ASSERT_FALSE(read_group_ids.contains(0) || read_group_ids.contains(1) || read_group_ids.contains(2));
}

TEST_F(LocalGroupTranslatorTest, testCreateNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    auto group_id_to_group_name = RelearnTypes::group_names{};
    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, number_neurons, mt);
    const auto neuron_id_to_group_ids_unordered = neuron_id_to_group_ids
                                                  | ranges::views::transform([](const RelearnTypes::group_ids& group_ids) {
                                                        return group_ids | ranges::to<std::unordered_set>;
                                                    })
                                                  | ranges::to_vector;

    const auto translator = std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);

    ASSERT_THROW_NO_PRINT(translator->create_neurons(0), RelearnException);

    const auto number_created_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    translator->create_neurons(number_created_neurons);

    const auto& group_id_to_group_name_new = translator->get_all_group_names();
    const auto& neuron_id_to_group_ids_new = translator->get_neuron_ids_to_group_ids();
    const auto& neuron_id_to_group_ids_unordered_new = translator->get_neuron_ids_to_group_ids_unordered();

    ASSERT_EQ(group_id_to_group_name, group_id_to_group_name_new);

    ASSERT_TRUE(neuron_id_to_group_ids_new.size() == neuron_id_to_group_ids.size() + number_created_neurons);
    ASSERT_TRUE(neuron_id_to_group_ids_unordered_new.size() == neuron_id_to_group_ids_unordered.size() + number_created_neurons);

    for (auto neuron_id = number_neurons; neuron_id < neuron_id_to_group_ids_new.size(); ++neuron_id) {
        ASSERT_EQ(neuron_id_to_group_ids_new[neuron_id], RelearnTypes::group_ids{ Constants::default_group_id });
        ASSERT_EQ(neuron_id_to_group_ids_unordered_new[neuron_id], RelearnTypes::group_ids_unordered{ Constants::default_group_id });
    }
}
