/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_background_activity_io.h"

#include "Types.h"

#include "io/BackgroundActivityIO.h"
#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/helper/ChoiceFunction.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "util/NeuronID.h"
#include "util/shuffle/shuffle.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"

#include <gtest/gtest.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {
template <typename Base, typename T>
inline bool instanceof(const std::shared_ptr<T>& ptr) {
    return dynamic_cast<Base*>(ptr.get()) != nullptr;
}
} // namespace

void check_choice(const std::vector<std::string>& input_keys, const std::unordered_set<size_t>& indices, const std::vector<std::string>& expected_input_keys) {
    ASSERT_EQ(indices.size(), expected_input_keys.size());

    for (const auto index : indices) {
        ASSERT_LT(index, input_keys.size());
        const auto& input_key = input_keys[index];
        ASSERT_TRUE(std::find(expected_input_keys.begin(), expected_input_keys.end(), input_key) != expected_input_keys.end());
    }
}

std::vector<std::string> inputs_to_input_keys(const std::vector<std::shared_ptr<ActivityInput>>& inputs) {
    auto input_keys = std::vector<std::string>{};
    input_keys.reserve(inputs.size());

    for (const auto& input : inputs) {
        std::string key = std::string{};

        if (instanceof<ConstantActivityInput>(input)) {
            const auto constant_input = static_pointer_cast<ConstantActivityInput>(input);
            key = fmt::format("constant:{}", constant_input->get_constant());
        } else if (instanceof<NormalActivityInput>(input)) {
            const auto constant_input = static_pointer_cast<NormalActivityInput>(input);
            key = fmt::format("normal:{},{}", constant_input->get_mean(), constant_input->get_stddev());
        } else if (instanceof<FastNormalActivityInput>(input)) {
            const auto constant_input = static_pointer_cast<FastNormalActivityInput>(input);
            key = fmt::format("fastnormal:{},{}", constant_input->get_mean(), constant_input->get_stddev());
        }
        const bool found_key = !key.empty();
        RelearnException::check(found_key, "BackgroundActivityIOTest::inputs_to_input_keys: Key was not found");

        input_keys.push_back(key);
    }
    return input_keys;
}

TEST_F(BackgroundActivityIOTest, testRead) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto my_rank = mpiPP::MPIRank::root_rank();

    std::filesystem::path file_path{ "./background_activity0.tmp" };

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    of << "1 4 constant:1 .+\n";
    of << "2 4 constant:2 .+\n";
    of << "3 4 constant:3 .+\n";
    of << "10 -1 normal:1,1 .+\n";
    of << "150 -1 constant:2 .+\n";
    of.close();

    auto group_names = RelearnTypes::group_names{};
    auto neuron_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    NeuronsFactory::generate_random_neuron_groups(neuron_to_group_ids, group_names, num_neurons, mt, std::nullopt, std::nullopt, 1);
    const auto local_group_translator = std::make_shared<LocalGroupTranslator>(group_names, neuron_to_group_ids);

    auto tup = BackgroundActivityIO::load_background_activity(file_path, my_rank, local_group_translator);
    const auto& inputs = tup.first;
    auto chooser = std::move(tup.second);
    const auto input_keys = inputs_to_input_keys(inputs);

    ASSERT_EQ(inputs.size(), 4);
    const auto neuron_id = NeuronID{ 0 };

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(0, neuron_id).size(), 0);

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(1, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(1, neuron_id), { "constant:1" });

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(2, neuron_id).size(), 2);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(2, neuron_id), { "constant:1", "constant:2" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(3, neuron_id).size(), 3);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(3, neuron_id), { "constant:1", "constant:2", "constant:3" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(4, neuron_id).size(), 0);
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(9, neuron_id).size(), 0);
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(10, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, neuron_id), { "normal:1,1" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(100, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(100, neuron_id), { "normal:1,1" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(150, neuron_id).size(), 2);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(150, neuron_id), { "normal:1,1", "constant:2" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(500, neuron_id).size(), 2);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(150, neuron_id), { "normal:1,1", "constant:2" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(1000000, neuron_id).size(), 2);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(150, neuron_id), { "normal:1,1", "constant:2" });

    std::filesystem::remove(file_path);
}
TEST_F(BackgroundActivityIOTest, testAllInputTypes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = 10;
    const auto my_rank = mpiPP::MPIRank::root_rank();

    std::filesystem::path file_path{ "./background_activity1.tmp" };
    std::ofstream of(file_path, std::ios::binary | std::ios::out);
    const auto is_good = of.good();
    const auto is_bad = of.bad();
    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    of << "1 10 constant:42 .+\n";
    of << "10 20 fastnormal:4,2 .+\n";
    of << "20 30 normal:9,3 .+\n";
    of.close();

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(num_neurons, mt);

    const auto& [inputs, chooser] = BackgroundActivityIO::load_background_activity(file_path, my_rank, local_group_translator);
    const auto input_keys = inputs_to_input_keys(inputs);

    ASSERT_EQ(inputs.size(), 3);
    const auto neuron_id = NeuronID{ 0 };

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(0, neuron_id).size(), 0);

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(1, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(1, neuron_id), { "constant:42" });

    ASSERT_EQ(chooser->get_inputs_for_neuron_id(10, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(2, neuron_id), { "fastnormal:4,2" });
    ASSERT_EQ(chooser->get_inputs_for_neuron_id(20, neuron_id).size(), 1);
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(3, neuron_id), { "normal:9,3" });
    std::filesystem::remove(file_path);
}

TEST_F(BackgroundActivityIOTest, testNeuronIds) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }
    const auto num_neurons = 10;
    const auto my_rank = mpiPP::MPIRank::root_rank();

    std::filesystem::path file_path{ "./background_activity2.tmp" };
    std::ofstream of(file_path, std::ios::binary | std::ios::out);
    const auto is_good = of.good();
    const auto is_bad = of.bad();
    ASSERT_TRUE(is_good);
    ASSERT_FALSE(is_bad);

    of << "0 -1 constant:42 0:1 0:2\n";
    of << "10 20 fastnormal:4,2 0:1 0:2 0:5\n";
    of << "20 30 normal:9,3 0:3 0:5\n";
    of.close();

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(num_neurons, mt);

    const auto& [inputs, chooser] = BackgroundActivityIO::load_background_activity(file_path, my_rank, local_group_translator);
    const auto input_keys = inputs_to_input_keys(inputs);

    ASSERT_EQ(inputs.size(), 3);

    check_choice(input_keys, chooser->get_inputs_for_neuron_id(0, NeuronID{ 0 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(0, NeuronID{ 1 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(0, NeuronID{ 2 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(0, NeuronID{ 3 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(0, NeuronID{ 4 }), {});

    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, NeuronID{ 0 }), { "constant:42", "fastnormal:4,2" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, NeuronID{ 1 }), { "constant:42", "fastnormal:4,2" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, NeuronID{ 2 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, NeuronID{ 3 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(10, NeuronID{ 4 }), { "fastnormal:4,2" });

    check_choice(input_keys, chooser->get_inputs_for_neuron_id(20, NeuronID{ 0 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(20, NeuronID{ 1 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(20, NeuronID{ 2 }), { "normal:9,3" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(20, NeuronID{ 3 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(20, NeuronID{ 4 }), { "normal:9,3" });

    check_choice(input_keys, chooser->get_inputs_for_neuron_id(30, NeuronID{ 0 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(30, NeuronID{ 1 }), { "constant:42" });
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(30, NeuronID{ 2 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(30, NeuronID{ 3 }), {});
    check_choice(input_keys, chooser->get_inputs_for_neuron_id(30, NeuronID{ 4 }), {});
    std::filesystem::remove(file_path);
}