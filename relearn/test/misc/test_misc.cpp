/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_misc.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/Accumulate.h"
#include "util/File.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/shuffle/shuffle.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <vector>

TEST_F(MiscTest, testMinMaxAccEmpty) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const double>{}, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const float>{}, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const int>{}, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const std::size_t>{}, {}), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeMismatch) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = 3;
    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(num_neurons);

    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const double>{ { 4.0, 1.2 } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const float>{ { 0.8F } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const int>{ { 5, -4, 8, -6, 9 } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const std::size_t>{ { 10, 422, 5223, 554315 } }, extra_infos->get_disable_flags()), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeAllDisabled) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = 3;
    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(num_neurons);

    const auto disabled_neurons = NeuronIDRange::range(num_neurons) | ranges::to_vector;
    extra_infos->set_disabled_neurons(disabled_neurons);

    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const double>{ { 4.0, 1.2, 5.2 } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const float>{ { 0.8F, -1.6F, 65423.8F } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const int>{ { 5, -4, 8 } }, extra_infos->get_disable_flags()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const std::size_t>{ { 10, 422, 5223 } }, extra_infos->get_disable_flags()), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeAllStatic) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto update_status = std::vector<UpdateStatus>(3, UpdateStatus::Static);

    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const double>{ { 4.0, 1.2, 5.2 } }, update_status), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const float>{ { 0.8F, -1.6F, 65423.8F } }, update_status), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const int>{ { 5, -4, 8 } }, update_status), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = Util::min_max_acc(std::span<const std::size_t>{ { 10, 422, 5223 } }, update_status), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccDouble) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_enabled = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_disabled = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_static = NeuronIdFactory::get_random_number_neurons(mt);

    const auto number_values = number_enabled + number_disabled + number_static;

    const auto num_enabled = static_cast<std::ptrdiff_t>(number_enabled);
    const auto num_disabled = static_cast<std::ptrdiff_t>(number_disabled);
    const auto num_static = static_cast<std::ptrdiff_t>(number_values - (number_disabled + number_enabled));

    const auto update_status = ranges::views::concat(
                                   ranges::views::repeat_n(UpdateStatus::Enabled, num_enabled),
                                   ranges::views::repeat_n(UpdateStatus::Disabled, num_disabled),
                                   ranges::views::repeat_n(UpdateStatus::Static, num_static))
                               | ranges::to_vector | actions::shuffle(mt);

    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_values);

    auto disabled_neurons = std::vector<NeuronID>{};
    auto static_neurons = std::vector<NeuronID>{};
    for (const auto& neuron_id : NeuronIDRange::range(number_values)) {
        const auto& us = update_status[neuron_id.get_neuron_id()];
        if (us == UpdateStatus::Static) {
            static_neurons.push_back(neuron_id);
        } else if (us == UpdateStatus::Disabled) {
            disabled_neurons.push_back(neuron_id);
        }
    }

    extra_infos->set_disabled_neurons(disabled_neurons);
    extra_infos->set_static_neurons(static_neurons);

    auto values = std::vector<double>{};
    values.reserve(number_values);

    auto min = std::numeric_limits<double>::max();
    auto max = -std::numeric_limits<double>::max();
    auto sum = 0.0;

    for (auto i : ranges::views::indices(number_values)) {
        const auto random_value = RandomFactory::get_random_double(-100000.0, 100000.0, mt);

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(std::span<const double>{ values }, extra_infos->get_disable_flags());

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_NEAR(sum, accumulated, eps);
    ASSERT_EQ(number_enabled, num);
}

TEST_F(MiscTest, testMinMaxAccSizet) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_enabled = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_disabled = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_static = NeuronIdFactory::get_random_number_neurons(mt);

    const auto number_values = number_enabled + number_disabled + number_static;

    const auto update_status = ranges::views::concat(
                                   ranges::views::repeat_n(UpdateStatus::Enabled, static_cast<std::ptrdiff_t>(number_enabled)),
                                   ranges::views::repeat_n(UpdateStatus::Disabled, static_cast<std::ptrdiff_t>(number_disabled)),
                                   ranges::views::repeat_n(UpdateStatus::Static, static_cast<std::ptrdiff_t>(number_static)))
                               | ranges::to_vector | actions::shuffle(mt);

    auto values = std::vector<std::size_t>{};
    values.reserve(number_values);

    auto min = std::numeric_limits<std::size_t>::max();
    auto max = std::numeric_limits<std::size_t>::min();
    auto sum = std::size_t{ 0 };

    for (const auto i : ranges::views::indices(number_values)) {
        const auto random_value = RandomFactory::get_random_integer<std::size_t>(std::numeric_limits<std::size_t>::min(), std::numeric_limits<std::size_t>::max(), mt);

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_values);

    auto disabled_neurons = std::vector<NeuronID>{};
    auto static_neurons = std::vector<NeuronID>{};
    for (const auto& neuron_id : NeuronIDRange::range(number_values)) {
        const auto& us = update_status[neuron_id.get_neuron_id()];
        if (us == UpdateStatus::Static) {
            static_neurons.push_back(neuron_id);
        } else if (us == UpdateStatus::Disabled) {
            disabled_neurons.push_back(neuron_id);
        }
    }

    extra_infos->set_disabled_neurons(disabled_neurons);
    extra_infos->set_static_neurons(static_neurons);

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(std::span<const std::size_t>{ values }, extra_infos->get_disable_flags());

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_EQ(sum, accumulated);
    ASSERT_EQ(number_enabled, num);
}

TEST_F(MiscTest, testFindFileForRank) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto write_to_file = [](auto path) {
        auto of = std::ofstream{ path };
        of << "# Hello";
    };

    const auto* const expected_path1 = "./hello002.txt";
    const auto* const expected_path2 = "./test0.txt";
    const auto* const expected_path3 = "./0";
    const auto* const expected_path4 = "./step_100000_rank_19201.txt";

    // These names are also asserted to NOT be found one directory up (see the ".." checks below).
    // A previous run of this very test, executed with a shallower working directory, can leave
    // stray copies behind that then sit in what is now this run's parent directory -- delete any
    // such leftovers up front so this test is self-cleaning instead of depending on the ambient
    // working tree being pristine.
    for (const auto* const name : { "hello002.txt", "test0.txt", "0", "step_100000_rank_19201.txt" }) {
        std::filesystem::remove(std::filesystem::path(".") / name);
        std::filesystem::remove(std::filesystem::path("..") / name);
    }

    struct CleanupGuard {
        ~CleanupGuard() {
            for (const auto* const name : { "hello002.txt", "test0.txt", "0", "step_100000_rank_19201.txt" }) {
                std::filesystem::remove(std::filesystem::path(".") / name);
            }
        }
    } cleanup_guard{};

    write_to_file(expected_path1);
    write_to_file(expected_path2);
    write_to_file(expected_path3);
    write_to_file(expected_path4);

    const auto directory = std::filesystem::path{ "." };

    const auto path1 = Util::find_file_for_rank(directory, mpiPP::MPIRank{ 2 }, "hello", ".txt");
    const auto path2 = Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "test", ".txt");
    const auto path3 = Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "", "");
    const auto path4 = Util::find_file_for_rank(directory, mpiPP::MPIRank{ 19201 }, "step_100000_rank_", ".txt");

    ASSERT_EQ(path1, std::filesystem::path(expected_path1));
    ASSERT_EQ(path2, std::filesystem::path(expected_path2));
    ASSERT_EQ(path3, std::filesystem::path(expected_path3));
    ASSERT_EQ(path4, std::filesystem::path(expected_path4));

    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank::uninitialized_rank(), "hello", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank::uninitialized_rank(), "test", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank::uninitialized_rank(), "", ""), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank::uninitialized_rank(), "step_100000_rank_", ".txt"), RelearnException);

    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "hello", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 1 }, "test", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 4 }, "", ""), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 52 }, "step_100000_rank_", ".txt"), RelearnException);

    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 2 }, "test", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "hello", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "test", ""), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 19201 }, "rank_", ".txt"), RelearnException);

    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 2 }, "hello", ".temp"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "test", ".tmp"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 0 }, "", ".t"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(directory, mpiPP::MPIRank{ 19201 }, "step_100000_rank_", ".text"), RelearnException);

    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(std::filesystem::path(".."), mpiPP::MPIRank{ 2 }, "hello", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(std::filesystem::path(".."), mpiPP::MPIRank{ 0 }, "test", ".txt"), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(std::filesystem::path(".."), mpiPP::MPIRank{ 0 }, "", ""), RelearnException);
    ASSERT_THROW_NO_PRINT(Util::find_file_for_rank(std::filesystem::path(".."), mpiPP::MPIRank{ 19201 }, "step_100000_rank_", ".txt"), RelearnException);
}
