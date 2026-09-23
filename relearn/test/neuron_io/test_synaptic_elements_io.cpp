/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synaptic_elements_io.h"

#include "io/SynapticElementsIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/view/zip.hpp>

#include <array>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <vector>

namespace {
/**
 * Flattens the calculators into the order in which the file spells the values,
 * so that they can be zipped against the gold data.
 */
[[nodiscard]] std::array<std::function<double(RelearnTypes::number_neurons_type)>, 15>
in_file_order(const SynapticElementsIO::Calculators& calculators) {
    const auto& [axons, den_exc, den_inh] = calculators;
    return {
        axons.min_calcium, axons.nu, axons.vacant_retract_ratio, axons.min_elements, axons.max_elements,
        den_exc.min_calcium, den_exc.nu, den_exc.vacant_retract_ratio, den_exc.min_elements, den_exc.max_elements,
        den_inh.min_calcium, den_inh.nu, den_inh.vacant_retract_ratio, den_inh.min_elements, den_inh.max_elements
    };
}
} // namespace

TEST_F(SynapticElementsIOTest, testRead) {
    const auto num_neurons = 25;
    const auto array_size = 15;
    auto group_names = RelearnTypes::group_names{};
    auto neuron_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_to_group_ids, group_names, num_neurons, mt);
    const auto local_group_translator = std::make_shared<LocalGroupTranslator>(group_names, neuron_to_group_ids);

    auto gold_data = std::vector<std::array<double, array_size>>{};

    auto ofs = std::ofstream("io0.tmp", std::ios::out);
    for (const auto neuron_id : NeuronIDRange::range(num_neurons)) {
        ofs << "0:" << neuron_id.get_neuron_id() + 1 << " ";
        auto gold = std::array<double, array_size>{ 0.0 };
        for (auto& value : gold) {
            value = RandomFactory::get_random_double(0.0, SynapticElementsIOTest::value_upper_limit, mt);
            ofs << value << " ";
        }
        ofs << "\n";
        gold_data.push_back(gold);
    }
    ofs.close();

    const auto loaded_calculators = SynapticElementsIO::load_function_from_file("io0.tmp", mpiPP::MPIRank::root_rank(), local_group_translator);
    const auto functions = in_file_order(loaded_calculators);

    for (const auto neuron_id : NeuronIDRange::range(num_neurons)) {
        for (const auto& [gold, func] : ranges::views::zip(gold_data[neuron_id.get_neuron_id()], functions)) {
            ASSERT_NEAR(gold, func(neuron_id.get_neuron_id()), eps);
        }
    }
    std::filesystem::remove("io0.tmp");
}

TEST_F(SynapticElementsIOTest, testReadDefault) {
    const auto num_neurons = 25;
    const auto array_size = 15;
    auto group_names = RelearnTypes::group_names{};
    auto neuron_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_to_group_ids, group_names, num_neurons, mt);
    const auto local_group_translator = std::make_shared<LocalGroupTranslator>(group_names, neuron_to_group_ids);
    const auto my_rank = mpiPP::MPIRank{ 3 };

    auto gold_data = std::vector<std::array<double, array_size>>{};

    auto ofs = std::ofstream("io1.tmp", std::ios::out);
    ofs << "default ";
    auto default_value = std::array<double, array_size>{ 0.0 };
    for (auto& value : default_value) {
        value = RandomFactory::get_random_double(0.0, SynapticElementsIOTest::value_upper_limit, mt);
        ofs << value << " ";
    }
    ofs << "\n";

    for (const auto neuron_id : NeuronIDRange::range(num_neurons)) {
        const auto mpi_rank = MPIRankFactory::get_random_mpi_rank(5, mt);
        ofs << mpi_rank.get_rank() << ":" << neuron_id.get_neuron_id() + 1 << " ";
        auto gold = std::array<double, array_size>{ 0.0 };
        for (auto& value : gold) {
            value = RandomFactory::get_random_double(0.0, SynapticElementsIOTest::value_upper_limit, mt);
            ofs << value << " ";
        }
        ofs << "\n";
        if (mpi_rank == my_rank) {
            gold_data.push_back(gold);
        } else {
            gold_data.push_back(default_value);
        }
    }
    ofs.close();

    const auto loaded_calculators = SynapticElementsIO::load_function_from_file("io1.tmp", my_rank, local_group_translator);
    const auto functions = in_file_order(loaded_calculators);

    for (const auto neuron_id : NeuronIDRange::range(num_neurons)) {
        for (const auto& [gold, func] : ranges::views::zip(gold_data[neuron_id.get_neuron_id()], functions)) {
            ASSERT_NEAR(gold, func(neuron_id.get_neuron_id()), eps);
        }
    }
    std::filesystem::remove("io1.tmp");
}

TEST_F(SynapticElementsIOTest, testReadGroupNames) {
    const auto num_neurons = 25;
    const auto array_size = 15;
    auto group_names = RelearnTypes::group_names{};
    auto neuron_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_to_group_ids, group_names, num_neurons, mt);
    const auto num_groups = group_names.size();
    const auto local_group_translator = std::make_shared<LocalGroupTranslator>(group_names, neuron_to_group_ids);
    const auto my_rank = mpiPP::MPIRank{ 3 };

    const auto gold_data = std::vector<std::array<double, array_size>>{};

    auto ofs = std::ofstream("io2.tmp", std::ios::out);
    ofs << "default ";
    auto default_value = std::array<double, array_size>{ 0.0 };
    for (auto& value : default_value) {
        value = RandomFactory::get_random_double(0.0, SynapticElementsIOTest::value_upper_limit, mt);
        ofs << value << " ";
    }
    ofs << "\n";

    const auto random_group_id = RandomFactory::get_random_integer<RelearnTypes::group_id>(0, num_groups - 1, mt);
    ofs << group_names[random_group_id] << " ";
    auto group_value = std::array<double, array_size>{ 0.0 };
    for (auto& value : group_value) {
        value = RandomFactory::get_random_double(0.0, SynapticElementsIOTest::value_upper_limit, mt);
        ofs << value << " ";
    }
    ofs << "\n";

    ofs.close();

    const auto loaded_calculators = SynapticElementsIO::load_function_from_file("io2.tmp", my_rank, local_group_translator);
    const auto functions = in_file_order(loaded_calculators);

    for (const auto neuron_id : NeuronIDRange::range_id(num_neurons)) {
        const auto& group_ids = neuron_to_group_ids[neuron_id];
        const auto contains_random_group_id = ranges::contains(group_ids, random_group_id);
        const auto& value = contains_random_group_id ? group_value : default_value;
        for (const auto& [val, func] : ranges::views::zip(value, functions)) {
            ASSERT_NEAR(val, func(neuron_id), eps);
        }
    }
    std::filesystem::remove("io2.tmp");
}