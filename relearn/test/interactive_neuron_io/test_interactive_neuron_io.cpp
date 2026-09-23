/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_interactive_neuron_io.h"

#include "RelearnTest.hpp"

#include "io/InteractiveNeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "factory/interval/interval_factory.h"
#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/indices.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {
void write_stimuli_to_file(std::filesystem::path path, std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>> stimuli) {
    auto of = std::ofstream(path, std::ios::binary | std::ios::out);

    for (const auto& [begin, end, frequency, intensity, names] : stimuli) {
        of << begin << "-" << end << ":" << frequency << " " << intensity;
        for (const auto& name : names) {
            of << " " << name;
        }
        of << '\n';
    }
}
} // namespace

TEST_F(InteractiveNeuronIOTest, testStimulusWithNeuronIds) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto number_neurons = local_group_translator->get_number_neurons_in_total();

    const auto path = std::filesystem::path("stimulus0.tmp");
    const auto number_ranks = RandomFactory::get_random_integer(2, 10, mt);
    const auto num_stimuli = RandomFactory::get_random_integer(1, 10, mt) * number_ranks;
    const auto number_steps = RandomFactory::get_random_integer(1000U, 100000U, mt);
    const auto my_rank = mpiPP::MPIRank(RandomFactory::get_random_integer(0, number_ranks - 1, mt));

    auto stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>>{};
    auto my_stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>>{};
    const auto intervals = IntervalFactory::get_random_non_overlapping_intervals(static_cast<unsigned int>(num_stimuli), number_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        const auto& interval = intervals[static_cast<std::size_t>(i)];
        const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);
        const auto& ids = NeuronIdFactory::get_random_neuron_ids(number_neurons, RandomFactory::get_random_integer(RelearnTypes::number_neurons_type{ 1 }, number_neurons, mt), mt);

        auto rank_ids = std::unordered_set<std::string>{};
        auto my_ids = std::unordered_set<NeuronID>{};

        for (const auto& neuron_id : ids) {
            const auto rank = mpiPP::MPIRank(RandomFactory::get_random_integer(0, number_ranks - 1, mt));
            rank_ids.insert(std::to_string(rank.get_rank()) + ":" + std::to_string(neuron_id.get_neuron_id() + 1));
            if (rank == my_rank) {
                my_ids.insert(neuron_id);
            }
        }

        stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, rank_ids);
        my_stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, my_ids);
    }

    write_stimuli_to_file(path, stimuli);

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, my_rank, local_group_translator);
    for (auto step = 0U; step < number_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);

        auto read_stimulated_neurons = size_t{ 0 };
        auto stimulated_neurons = size_t{ 0 };
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();

            auto found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step && my_ids == neuron_ids) {
                    // Same stimulus
                    stimulated_neurons += my_ids.size();
                    found_my_stimuli = true;
                    ASSERT_NEAR(my_intensity, intensity, eps);
                    break;
                }
            }

            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }

    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testStimulusWithGroups) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto& group_names = local_group_translator->get_all_group_names();

    const auto path = std::filesystem::path("stimulus1.tmp");
    const auto num_rank = RandomFactory::get_random_integer(2, 10, mt);
    const auto num_stimuli = RandomFactory::get_random_integer(1, 10, mt) * num_rank;
    const auto number_steps = RandomFactory::get_random_integer(1000U, 100000U, mt);
    const auto my_rank = mpiPP::MPIRank(RandomFactory::get_random_integer(0, num_rank - 1, mt));

    auto stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>>{};
    auto my_stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>>{};
    const auto intervals = IntervalFactory::get_random_non_overlapping_intervals(static_cast<unsigned int>(num_stimuli), number_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        const auto& interval = intervals[static_cast<std::size_t>(i)];

        const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);
        const auto chosen_group_names = RandomFactory::sample(group_names, mt);

        auto my_ids = std::unordered_set<NeuronID>{};
        for (const auto& group_name : chosen_group_names) {
            for (const auto& neuron_id : local_group_translator->get_neuron_ids_in_group(local_group_translator->get_group_id_for_group_name(group_name))) {
                my_ids.insert(neuron_id);
            }
        }

        stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, chosen_group_names | ranges::to<std::unordered_set>);
        my_stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, my_ids);
    }

    write_stimuli_to_file(path, stimuli);

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, my_rank, local_group_translator);
    for (auto step = 0U; step < number_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);

        auto read_stimulated_neurons = size_t{ 0 };
        auto stimulated_neurons = size_t{ 0 };
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();

            auto found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step && my_ids == neuron_ids) {
                    // Same stimulus
                    stimulated_neurons += my_ids.size();
                    found_my_stimuli = true;
                    ASSERT_NEAR(my_intensity, intensity, eps);
                    break;
                }
            }

            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testFrequency) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto number_neurons = local_group_translator->get_number_neurons_in_total();
    const auto path = std::filesystem::path("stimulus2.tmp");
    const auto number_steps = RandomFactory::get_random_integer(20000U, 100000U, mt);

    auto intervals = std::vector<utility::Interval<RelearnTypes::step_type>>{};

    const auto begin = RandomFactory::get_random_integer(0U, number_steps - 10000, mt);
    const auto end = RandomFactory::get_random_integer(begin + 50, number_steps, mt);
    const auto frequency = RandomFactory::get_random_integer(2U, 10U, mt);

    const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);
    const auto& ids = NeuronIdFactory::get_random_neuron_ids(number_neurons, RandomFactory::get_random_integer(RelearnTypes::number_neurons_type{ 1 }, number_neurons, mt), mt);
    auto rank_ids = std::unordered_set<std::string>{};
    for (const auto& neuron_id : ids) {
        rank_ids.insert("0:" + std::to_string(neuron_id.get_neuron_id() + 1));
    }

    write_stimuli_to_file(path, { std::make_tuple(begin, end, frequency, intensity, rank_ids) });

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator);
    for (auto step = 0U; step < number_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);

        if (step < begin || step > end || (step - begin) % frequency != 0) {
            ASSERT_EQ(0, read_stimuli.size());
        } else {
            ASSERT_EQ(1, read_stimuli.size());
            const auto& [read_neuron_ids, read_intensity] = read_stimuli[0];
            ASSERT_EQ(ids, read_neuron_ids);
            ASSERT_FALSE(read_neuron_ids.empty());
            ASSERT_NEAR(intensity, read_intensity, eps);
        }
    }
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testEmptyNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto number_neurons = local_group_translator->get_number_neurons_in_total();
    const auto path = std::filesystem::path("stimulus3.tmp");
    const auto num_stimuli = RandomFactory::get_random_integer(2, 10, mt);
    const auto number_steps = RandomFactory::get_random_integer(1000U, 100000U, mt);

    auto stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<std::string>>>{};
    auto my_stimuli = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, RelearnTypes::step_type, double, std::unordered_set<NeuronID>>>{};
    const auto intervals = IntervalFactory::get_random_non_overlapping_intervals(static_cast<unsigned int>(num_stimuli), number_steps, mt);

    for (const auto i : ranges::views::indices(num_stimuli)) {
        const auto interval = intervals[static_cast<std::size_t>(i)];
        const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);

        const auto& ids = (i % 2 == 0) ? NeuronIdFactory::get_random_neuron_ids(number_neurons, RandomFactory::get_random_integer(RelearnTypes::number_neurons_type{ 1 }, number_neurons, mt), mt) : std::unordered_set<NeuronID>{};

        auto rank_ids = std::unordered_set<std::string>{};
        for (const auto& neuron_id : ids) {
            rank_ids.insert("0:" + std::to_string(neuron_id.get_neuron_id() + 1));
        }

        stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, rank_ids);
        my_stimuli.emplace_back(interval.begin, interval.end, 1U, intensity, ids);
    }

    write_stimuli_to_file(path, stimuli);

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator);
    for (auto step = 0U; step < number_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);

        auto read_stimulated_neurons = size_t{ 0 };
        auto stimulated_neurons = size_t{ 0 };
        for (const auto& [neuron_ids, intensity] : read_stimuli) {
            read_stimulated_neurons += neuron_ids.size();

            auto found_my_stimuli = false;
            for (const auto& [begin, end, frequency, my_intensity, my_ids] : my_stimuli) {
                if (begin <= step && end >= step && my_ids == neuron_ids) {
                    // Same stimulus
                    stimulated_neurons += my_ids.size();
                    found_my_stimuli = true;
                    ASSERT_NEAR(my_intensity, intensity, eps);
                    break;
                }
            }

            ASSERT_TRUE(found_my_stimuli);
        }

        ASSERT_EQ(stimulated_neurons, read_stimulated_neurons);
    }
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testNoFile) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto path = std::filesystem::path("stimulus4.tmp");
    ASSERT_THROW_NO_PRINT(std::ignore = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator), RelearnException);
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testEmptyFile) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto path = std::filesystem::path("stimulus5.tmp");

    write_stimuli_to_file(path, {});

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator);
    for (auto step = 0U; step < 1000; step++) {
        const auto& read_stimuli = stimulus_function(step);
        ASSERT_EQ(0, read_stimuli.size());
    }
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testInvalidNeuronId) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto number_neurons = local_group_translator->get_number_neurons_in_total();
    const auto path = std::filesystem::path("stimulus6.tmp");
    const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);
    const auto rank_ids = std::unordered_set<std::string>{ "0:" + std::to_string(number_neurons + 1) };
    const auto number_steps = RandomFactory::get_random_integer(1000U, 100000U, mt);
    const auto interval = IntervalFactory::get_random_interval(number_steps, 1U, mt);

    write_stimuli_to_file(path, { std::make_tuple(interval.begin, interval.end, 1U, intensity, rank_ids) });

    ASSERT_THROW_NO_PRINT(std::ignore = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator), RelearnException);
    std::filesystem::remove(path);
}

TEST_F(InteractiveNeuronIOTest, testInvalidGroupName) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local_group_translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);
    const auto path = std::filesystem::path("stimulus7.tmp");

    const auto invalid_group_name = NeuronsFactory::get_invalid_group_name(local_group_translator->get_all_group_names(), mt);
    const auto number_steps = RandomFactory::get_random_integer(1000U, 100000U, mt);
    const auto intensity = RandomFactory::get_random_double(0.001, 100.0, mt);
    const auto interval = IntervalFactory::get_random_interval(number_steps, 1U, mt);

    write_stimuli_to_file(path, { std::make_tuple(interval.begin, interval.end, 1U, intensity, std::unordered_set{ invalid_group_name }) });

    const auto stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(path, mpiPP::MPIRank::root_rank(), local_group_translator);
    for (auto step = 0U; step < number_steps; step++) {
        const auto& read_stimuli = stimulus_function(step);
        ASSERT_EQ(0, read_stimuli.size());
    }
    std::filesystem::remove(path);
}
