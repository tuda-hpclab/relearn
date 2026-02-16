/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_monitor_parser.h"

#include "Config.h"
#include "Types.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/shuffle/shuffle.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/helper/RankNeuronIdAdapter.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/generate.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

TEST_F(MonitorParserTest, testParseIds) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto my_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

    auto my_number_neurons = std::size_t{ 0 };

    const auto random_number_neurons = [this]() { return NeuronIdFactory::get_random_number_neurons(mt); };

    const auto create_rank_neuron_ids = [my_rank, &my_number_neurons](const auto& rank_num_neurons_pair) {
        const auto& rank = std::get<0>(rank_num_neurons_pair);
        const auto& number_neurons = std::get<1>(rank_num_neurons_pair);

        if (rank == my_rank) {
            my_number_neurons = number_neurons;
        }

        return NeuronID::range(number_neurons)
               | ranges::views::transform([rank](const NeuronID& neuron_id) -> RankNeuronId { return { rank, neuron_id }; });
    };

    const auto rank_neuron_ids = ranges::views::zip(
                                     mpiPP::MPIRank::range(number_ranks),
                                     ranges::views::generate(random_number_neurons))
                                 | ranges::views::for_each(create_rank_neuron_ids)
                                 | ranges::to_vector
                                 | actions::shuffle(mt);

    auto ss = std::stringstream{};
    ss << "0:1";

    for (const auto& rni : rank_neuron_ids) {
        if (rni.get_rank() != my_rank) {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
            continue;
        }

        const auto use_default = RandomFactory::get_random_bool(mt);

        if (use_default) {
            ss << ";-1:" << rni.get_neuron_id().get_neuron_id() + 1;
        } else {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
        }
    }

    auto translator = std::make_shared<LocalGroupTranslator>(RelearnTypes::group_names({ "random" }), std::vector<RelearnTypes::group_ids>({ { 0 } }));

    const auto& parsed_ids = MonitorParser::parse_my_ids(ss.str(), my_rank, translator);

    ASSERT_EQ(parsed_ids.size(), my_number_neurons);

    for (const auto neuron_id : NeuronID::range_id(my_number_neurons)) {
        ASSERT_EQ(parsed_ids[neuron_id], NeuronID(neuron_id));
    }
}

TEST_F(MonitorParserTest, testParseGroups) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    const auto& group_names = NeuronsFactory::get_random_group_names_specific(3, mt);
    const auto& group_ids = NeuronsFactory::get_random_group_ids({ group_names.size(), number_neurons }, mt);
    auto translator = std::make_shared<LocalGroupTranslator>(group_names, group_ids);

    const auto str = group_names[1] + ";" + group_names[2] + ";" + NeuronsFactory::get_random_group_name(mt);

    const auto& parsed_ids = MonitorParser::parse_my_ids(str, mpiPP::MPIRank{ 0 }, translator);

    const auto num_neurons_in_groups = translator->get_number_neurons_in_group(1) + translator->get_number_neurons_in_group(2);
    ASSERT_TRUE(parsed_ids.size() <= num_neurons_in_groups); // parsed_ids has no duplicates but since neurons can be in multiple groups num_neurons_in_groups could count some neurons multiple times

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto iter = std::ranges::find(parsed_ids, NeuronID{ neuron_id });

        ASSERT_TRUE((!ranges::contains(group_ids[neuron_id], 1) && !ranges::contains(group_ids[neuron_id], 2) && iter == parsed_ids.end()) || ((ranges::contains(group_ids[neuron_id], 1) || ranges::contains(group_ids[neuron_id], 2)) && iter != parsed_ids.end()));
    }
}

TEST_F(MonitorParserTest, testParseGroups2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    const auto& group_names = NeuronsFactory::get_random_group_names_specific(3, mt);
    const auto& group_ids = NeuronsFactory::get_random_group_ids({ group_names.size(), number_neurons }, mt, 1); // have maximum one group per neuron
    auto translator = std::make_shared<LocalGroupTranslator>(group_names, group_ids);

    const auto str = group_names[1] + ";" + group_names[2] + ";" + NeuronsFactory::get_random_group_name(mt);

    const auto& parsed_ids = MonitorParser::parse_my_ids(str, mpiPP::MPIRank{ 0 }, translator);

    const auto num_neurons_in_groups = translator->get_number_neurons_in_group(1) + translator->get_number_neurons_in_group(2);
    ASSERT_TRUE(parsed_ids.size() <= num_neurons_in_groups); // parsed_ids has no duplicates but since neurons can be in multiple groups num_neurons_in_groups could count some neurons multiple times

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto iter = std::ranges::find(parsed_ids, NeuronID{ neuron_id });

        ASSERT_TRUE((!ranges::contains(group_ids[neuron_id], 1) && !ranges::contains(group_ids[neuron_id], 2) && iter == parsed_ids.end()) || ((ranges::contains(group_ids[neuron_id], 1) || ranges::contains(group_ids[neuron_id], 2)) && iter != parsed_ids.end()));
    }
}

TEST_F(MonitorParserTest, testParseDefaultGroup) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    const auto& group_names = NeuronsFactory::get_random_group_names_specific(0, mt);
    const auto& group_ids = NeuronsFactory::get_random_group_ids({ group_names.size(), number_neurons }, mt); // every neuron in default group
    auto translator = std::make_shared<LocalGroupTranslator>(group_names, group_ids);

    const auto str = group_names[Constants::default_group_id];

    const auto& parsed_ids = MonitorParser::parse_my_ids(str, mpiPP::MPIRank{ 0 }, translator);

    const auto num_neurons_in_default = translator->get_number_neurons_in_group(Constants::default_group_id);

    ASSERT_EQ(parsed_ids.size(), num_neurons_in_default);

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto iter = std::ranges::find(parsed_ids, NeuronID{ neuron_id });

        ASSERT_TRUE(ranges::contains(group_ids[neuron_id], Constants::default_group_id) && iter != parsed_ids.end());
    }
}

TEST_F(MonitorParserTest, testParseGroupsRegex) {
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

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 15;
    const auto num_regex_groups = RandomFactory::get_random_integer(0U, 10U, mt);

    const auto max_groups_except_default = 10;
    auto group_names = NeuronsFactory::get_random_group_names(max_groups_except_default, mt);
    const auto random_groups = group_names.size();
    for (const auto& name : ranges::views::take(possible_group_names, num_regex_groups)) {
        group_names.emplace_back(name);
    }

    const auto num_groups = group_names.size();
    const auto& group_ids = NeuronsFactory::get_random_group_ids({ num_groups, number_neurons }, mt);
    auto translator = std::make_shared<LocalGroupTranslator>(group_names, group_ids);

    auto regex_neurons = 0ULL;
    for (auto group_id = random_groups; group_id < group_names.size(); group_id++) {
        regex_neurons += translator->get_number_neurons_in_group(group_id);
    }

    const std::string str = "REG.+EX";
    const auto& parsed_ids = MonitorParser::parse_my_ids(str, mpiPP::MPIRank{ 0 }, translator);

    ASSERT_TRUE(parsed_ids.size() <= regex_neurons); // because regex_neurons counts single neurons multiple times if they are in multiple groups

    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        const auto iter = std::ranges::find(parsed_ids, NeuronID{ neuron_id });

        auto has_regex_group = false;
        for (const auto& group_id : group_ids[neuron_id]) {
            if (group_id >= random_groups) {
                has_regex_group = true;
                break;
            }
        }
        if (has_regex_group) {
            ASSERT_TRUE(iter != parsed_ids.end());
        } else {
            ASSERT_TRUE(iter == parsed_ids.end());
        }
    }
}
