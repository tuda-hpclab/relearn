/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuronid_parser.h"

#include "io/parser/NeuronIdParser.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/rank_neuron_id/rank_neuron_id_factory.h"

#include <gtest/gtest.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <tuple>
#include <vector>

TEST_F(NeuronIdParserTest, testParseDescriptionFixed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto checker = [](std::string_view description, mpiPP::MPIRank rank, NeuronID::value_type neuron_id) {
        const auto opt_rni = NeuronIdParser::parse_description(description, rank);
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        const auto rni = RankNeuronId{ rank, NeuronID(neuron_id) };
        ASSERT_EQ(rni, parsed_rni);
    };

    checker("0:1", mpiPP::MPIRank(0), 0);
    checker("2:1", mpiPP::MPIRank(2), 0);
    checker("155:377", mpiPP::MPIRank(155), 376);
    checker("-1:17", mpiPP::MPIRank(5), 16);
}

TEST_F(NeuronIdParserTest, testUninitRank) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto val1 = NeuronIdParser::parse_description("", mpiPP::MPIRank());
    ASSERT_FALSE(val1.has_value());

    const auto val2 = NeuronIdParser::parse_description("155:377", mpiPP::MPIRank());
    ASSERT_FALSE(val2.has_value());

    const auto val3 = NeuronIdParser::parse_description("-1:17", mpiPP::MPIRank());
    ASSERT_FALSE(val3.has_value());
}

TEST_F(NeuronIdParserTest, testParseDescriptionException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto checker = [](std::string_view description, mpiPP::MPIRank default_rank) {
        const auto opt_rni = NeuronIdParser::parse_description(description, default_rank);
        ASSERT_FALSE(opt_rni.has_value());
    };

    checker("0:1:0", mpiPP::MPIRank::root_rank());
    checker("5:-4", mpiPP::MPIRank::root_rank());
    checker("+0:1", mpiPP::MPIRank::root_rank());
    checker("AB:1", mpiPP::MPIRank::root_rank());
    checker("-5:2", mpiPP::MPIRank::root_rank());
    checker("0:", mpiPP::MPIRank::root_rank());
    checker("5;2", mpiPP::MPIRank::root_rank());
    checker("", mpiPP::MPIRank::root_rank());
}

TEST_F(NeuronIdParserTest, testParseDescriptionException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto checker = [](std::string_view description, mpiPP::MPIRank default_rank) {
        ASSERT_THROW_NO_PRINT(std::ignore = NeuronIdParser::parse_description(description, default_rank);, RelearnException);
    };

    checker("0:0", mpiPP::MPIRank::root_rank());
    checker("1:0", mpiPP::MPIRank::root_rank());
    checker("-1:0", mpiPP::MPIRank::root_rank());
    checker("24575:0", mpiPP::MPIRank::root_rank());
}

TEST_F(NeuronIdParserTest, testParseDescriptionRandom) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    for (auto i = 0; i < 10000; i++) {
        const auto& [rni, descr] = RankNeuronIdFactory::generate_random_rank_neuron_id_description(mt);

        const auto opt_rni = NeuronIdParser::parse_description(descr, mpiPP::MPIRank(0));
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        ASSERT_EQ(rni, parsed_rni);
    }
}

TEST_F(NeuronIdParserTest, testParseDescriptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto rank_neuron_ids = std::vector<RankNeuronId>{};
    rank_neuron_ids.reserve(number_neurons + 1);

    auto ss = std::stringstream{};

    const auto& [first_rni, first_description] = RankNeuronIdFactory::generate_random_rank_neuron_id_description(mt);
    rank_neuron_ids.emplace_back(first_rni);

    ss << first_description;

    for (auto i = 0U; i < number_neurons; i++) {
        const auto& [new_rni, new_description] = RankNeuronIdFactory::generate_random_rank_neuron_id_description(mt);
        ss << ';' << new_description;

        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto& parsed_rnis = NeuronIdParser::parse_multiple_description(ss.str(), mpiPP::MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis);
}

TEST_F(NeuronIdParserTest, testParseDescriptionsFixed) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto rank_neuron_ids = std::vector<RankNeuronId>{
        { mpiPP::MPIRank(2), NeuronID(100) },
        { mpiPP::MPIRank(5), NeuronID(6) },
        { mpiPP::MPIRank(0), NeuronID(122) },
        { mpiPP::MPIRank(2), NeuronID(100) },
        { mpiPP::MPIRank(1674), NeuronID(1) },
        { mpiPP::MPIRank(89512), NeuronID(6) },
        { mpiPP::MPIRank(0), NeuronID(1) },
        { mpiPP::MPIRank(0), NeuronID(1) },
    };

    constexpr auto description_1 = "2:101;5:7;0:123;2:101;1674:2;89512:7;0:2;0:2";
    constexpr auto description_2 = "2:101;-1:7;0:123;2:101;1674:2;89512:7;0:2;0:2";
    constexpr auto description_3 = "2:101;5:7;-1:123;2:101;1674:2;89512:7;-1:2;0:2";
    constexpr auto description_4 = "2:101;5:7;-1:123;-8:801;2:101;6:;1674:2;-999:6;89512:7;-1:2;0:2";

    const auto& parsed_rnis_1 = NeuronIdParser::parse_multiple_description(description_1, mpiPP::MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_1);

    const auto& parsed_rnis_2 = NeuronIdParser::parse_multiple_description(description_2, mpiPP::MPIRank(5));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_2);

    const auto& parsed_rnis_3 = NeuronIdParser::parse_multiple_description(description_3, mpiPP::MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_3);

    const auto& parsed_rnis_4 = NeuronIdParser::parse_multiple_description(description_4, mpiPP::MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_4);
}

TEST_F(NeuronIdParserTest, testExtractNeuronIDs) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto rank_neuron_ids = std::vector<RankNeuronId>{};
    rank_neuron_ids.reserve(number_neurons + 2);

    const auto my_rank = MPIRankFactory::get_random_mpi_rank(100, mt);

    for (auto i = 0U; i < number_neurons; i++) {
        const auto& [new_rni, _] = RankNeuronIdFactory::generate_random_rank_neuron_id_description(mt);
        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto position_1 = RandomFactory::get_random_integer<std::ptrdiff_t>(0, static_cast<long int>(number_neurons), mt);
    const auto position_2 = RandomFactory::get_random_integer<std::ptrdiff_t>(0, static_cast<long int>(number_neurons), mt);

    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_1, RankNeuronId(my_rank, NeuronID(42)));
    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_2, RankNeuronId(my_rank, NeuronID(9874)));

    const auto golden_ids = rank_neuron_ids | ranges::views::filter(utility::equal_to(my_rank), &RankNeuronId::get_rank) | ranges::views::transform([](const RankNeuronId& rni) {
                                const auto& [rank, id] = rni;
                                return NeuronID(id.get_neuron_id());
                            })
                            | ranges::to_vector;

    const auto& extracted_ids = NeuronIdParser::extract_my_ids(rank_neuron_ids, my_rank);

    ASSERT_EQ(golden_ids, extracted_ids);
}

TEST_F(NeuronIdParserTest, testRemoveAndSort) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto neuron_ids = std::vector<NeuronID>{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0U; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdFactory::get_random_neuron_id(number_neurons, mt));
    }

    const auto& unique_and_filtered = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);

    for (auto i = 0U; i < unique_and_filtered.size() - 1; i++) {
        ASSERT_LE(unique_and_filtered[i].get_neuron_id(), unique_and_filtered[i + 1].get_neuron_id());
    }

    for (const auto& original_id : neuron_ids) {
        const auto pos = std::ranges::find(unique_and_filtered, original_id);
        ASSERT_NE(pos, unique_and_filtered.end());
    }

    for (const auto& new_id : unique_and_filtered) {
        const auto pos = std::ranges::find(neuron_ids, new_id);
        ASSERT_NE(pos, neuron_ids.end());
    }
}

TEST_F(NeuronIdParserTest, testRemoveAndSortException1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto neuron_ids = std::vector<NeuronID>{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0U; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdFactory::get_random_neuron_id(number_neurons, mt));
    }

    const auto virtual_rma = RandomFactory::get_random_integer<NeuronID::value_type>(0, 100000, mt);
    const auto position = RandomFactory::get_random_integer<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(number_neurons), mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID(true, virtual_rma));

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

TEST_F(NeuronIdParserTest, testRemoveAndSortException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto neuron_ids = std::vector<NeuronID>{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0U; i < number_neurons; i++) {
        neuron_ids.emplace_back(
            NeuronIdFactory::get_random_neuron_id(number_neurons, mt));
    }

    const auto position = RandomFactory::get_random_integer<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(number_neurons), mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID{});

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIdParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}
